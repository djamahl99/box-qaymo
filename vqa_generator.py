import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.waymo_loader import WaymoDatasetLoader
from waymovqa.prompts import get_all_prompt_generators, get_prompt_generator, BasePromptGenerator
from waymovqa.models import MODEL_REGISTRY
from waymovqa.models.base import BaseModel

from waymovqa.data.vqa_dataset import VQADataset

class VQAGenerator:
    """Main class for generating VQA samples."""

    def __init__(
        self,
        dataset_path: Path,
        prompt_generators: Optional[Union[List[str], List[BasePromptGenerator]]] = None,
        model: BaseModel = None
    ):
        """
        Initialize VQA generator.

        Args:
            prompt_generators: Optional list of prompt generator names or instances.
                               If None, all registered generators will be used.
        """
        self.dataset_path = Path(dataset_path)
        self.loader = WaymoDatasetLoader(self.dataset_path)
        self.dataset = VQADataset(tag='ground_truth')

        print("Registered prompt generators:", get_all_prompt_generators())
        if prompt_generators is None:
            # Use all registered prompt generators
            self.prompt_generators = [
                cls() for cls in get_all_prompt_generators().values()
            ]
        else:
            self.prompt_generators = []
            for gen in prompt_generators:
                if isinstance(gen, str):
                    # Get the generator by name
                    self.prompt_generators.append(get_prompt_generator(gen)())
                elif isinstance(gen, BasePromptGenerator):
                    # Use the provided instance
                    self.prompt_generators.append(gen)
                else:
                    raise TypeError(
                        f"Expected str or BasePromptGenerator, got {type(gen)}"
                    )
                    
    def find_scenes(self) -> List[str]:
        """Find scenes to sample from."""
        return self.loader.get_scene_ids()

    def find_scene_objects(self, scene_id: str) -> List[ObjectInfo]:
        """Find all ObjectInfo for a scene."""
        # First try to load from scene object table
        try:
            object_table = self.loader.load_scene_object_table(scene_id)
            if object_table:
                # Convert table entries to ObjectInfo objects
                objects = []
                for entry in object_table:
                    try:
                        obj = self.loader.load_object(entry["id"], scene_id)
                        objects.append(obj)
                    except FileNotFoundError:
                        continue
                return objects
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
        # Fallback: load objects by scanning object_infos directory
        object_pattern = f"object_*_{scene_id}.json"
        object_files = list(self.loader.object_infos_path.glob(object_pattern))
        
        objects = []
        for obj_file in object_files:
            try:
                obj = ObjectInfo.load(obj_file)
                objects.append(obj)
            except (json.JSONDecodeError, KeyError):
                continue
                
        return objects

    def generate_dataset_scene_based(self, questions_per_scene: int = 5) -> List[Dict[str, Any]]:
        """
        Generate VQA dataset based on scenes.

        Args:
            questions_per_scene: Number of questions to generate per scene

        Returns:
            List of VQA samples
        """
        raise NotImplementedError()

    def generate_dataset_object_based(self, total_samples: int = 500) -> List[Dict[str, Any]]:
        """
        Generate VQA dataset by sampling objects.

        Args:
            total_samples: Total number of samples to generate

        Returns:
            List of VQA samples
        """
        object_ids = self.loader.load_all_objects_with_cvat_ids()
        
        all_samples = []
        
        # First pass: generate a large pool of samples
        while len(all_samples) < total_samples * 3:
            # Sample an object id
            object_id = random.choice(object_ids)

            print('sampled', object_id)

            # get the corresponding scene
            scene_id = self.loader.get_object_id_to_scene_id()[object_id]

            try:
                # Load scene
                scene = self.loader.load_scene(scene_id)
                
                # Load object
                sampled_object = self.loader.load_object(object_id, scene_id)

                # Sample a frame for this object TODO: allow for multiple timestamp - integrate different answer / question in generator
                print('len(sampled_object.frames)', len(sampled_object.frames))
                timestamp = random.choice(sampled_object.frames)

                print('sampled timestamp', timestamp)
                frame = self.loader.load_frame(scene_id, timestamp)
                
                if not object:
                    continue

                # Generate samples
                for generator in self.prompt_generators:
                    samples = generator.generate(scene, [sampled_object], frame)
                    print('samples', samples)
                    all_samples.extend(samples)

            except FileNotFoundError:
                continue
                
            # Stop if we have enough samples
            if len(all_samples) >= total_samples * 3:  # Generate 3x to allow for selection
                break
        
        # TODO: sample analytics and then subsampling?
        subsamples = random.sample(all_samples, total_samples)

        for sample in subsamples:
            self.dataset.add_sample(*sample)

    def generate_dataset(self, questions_per_scene: int = 5, total_samples: int = 500, 
                         method: str = "combined") -> List[Dict[str, Any]]:
        """
        Generate VQA dataset using the specified method.

        Args:
            questions_per_scene: Number of questions per scene for scene-based generation
            total_samples: Total number of samples for object-based generation
            method: Generation method - "scene", "object", or "combined"

        Returns:
            List of VQA samples
        """
        if method == "scene":
            self.generate_dataset_scene_based(questions_per_scene)
        elif method == "object":
            self.generate_dataset_object_based(total_samples)
        elif method == "combined":
            # Generate half using scene-based approach and half using object-based
            self.generate_dataset_scene_based(questions_per_scene)
                
            self.generate_dataset_object_based(total_samples // 2)
            
            # TODO: shuffle
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'scene', 'object', or 'combined'")


def main():
    """Main function to generate and save a VQA dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VQA dataset from processed data")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed waymo dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--method", type=str, default="combined", choices=["scene", "object", "combined"],
                        help="Generation method")
    parser.add_argument("--questions_per_scene", type=int, default=5, 
                        help="Number of questions per scene for scene-based generation")
    parser.add_argument("--total_samples", type=int, default=500,
                        help="Total number of samples for object-based generation")
    parser.add_argument("--generators", type=str, nargs="+", default=None,
                        help="Specific prompt generators to use (by name)")
    parser.add_argument("--analyze", action="store_true", help="Analyze generated dataset")
    
    args = parser.parse_args()

    
    model = MODEL_REGISTRY[args.model.lower()]

    # Initialize generator
    generator = VQAGenerator(
        dataset_path=args.dataset_path,
        prompt_generators=args.generators,
        model=model
    )
    
    # Generate dataset
    generator.generate_dataset(
        questions_per_scene=args.questions_per_scene,
        total_samples=args.total_samples,
        method=args.method
    )
    
    output_path = Path(args.save_path)
    
    # Save dataset
    generator.dataset.save_dataset(str(output_path))


if __name__ == "__main__":
    main()