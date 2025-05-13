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
from waymovqa.waymo_loader import DatasetLoader
from waymovqa.prompts import get_all_prompt_generators, get_prompt_generator, BasePromptGenerator

class VQAGenerator:
    """Main class for generating VQA samples."""

    def __init__(
        self,
        dataset_path: Path,
        prompt_generators: Optional[Union[List[str], List[BasePromptGenerator]]] = None,
    ):
        """
        Initialize VQA generator.

        Args:
            prompt_generators: Optional list of prompt generator names or instances.
                               If None, all registered generators will be used.
        """
        self.dataset_path = Path(dataset_path)
        self.loader = DatasetLoader(self.dataset_path)

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
        dataset = []
        
        # Get all scenes
        scene_ids = self.find_scenes()
        
        for scene_id in scene_ids:
            print(f"Processing scene {scene_id}...")
            
            try:
                # Load scene
                scene = self.loader.load_scene(scene_id)
                
                # Load objects for this scene
                objects = self.find_scene_objects(scene_id)
                
                if not objects:
                    print(f"No objects found for scene {scene_id}, skipping")
                    continue
                
                # Get timestamps for this scene
                timestamps = set()
                for obj in objects:
                    if hasattr(obj, "frames") and obj.frames:
                        timestamps.update(obj.frames)
                
                if not timestamps:
                    print(f"No timestamps found for scene {scene_id}, skipping")
                    continue
                
                timestamps = sorted(list(timestamps))
                
                # Sample a few timestamps to focus on
                sampled_timestamps = random.sample(
                    timestamps, 
                    min(3, len(timestamps))
                )
                
                scene_samples = []
                
                # Generate questions for each sampled timestamp
                for timestamp in sampled_timestamps:
                    try:
                        # Load frame
                        frame = self.loader.load_frame(scene_id, timestamp)
                        
                        # Get objects visible in this frame
                        visible_objects = [obj for obj in objects if timestamp in obj.frames]
                        
                        # Generate questions for each prompt generator
                        for generator in self.prompt_generators:
                            samples = generator.generate(scene, visible_objects, frame)
                            scene_samples.extend(samples)
                    except FileNotFoundError:
                        continue
                
                # Sample questions to include in the final dataset
                if scene_samples:
                    # Ensure we have at least one sample from each generator if possible
                    question_types = {sample["question_type"] for sample in scene_samples}
                    
                    # First select one sample of each type
                    selected_samples = []
                    for q_type in question_types:
                        type_samples = [s for s in scene_samples if s["question_type"] == q_type]
                        if type_samples:
                            selected_samples.append(random.choice(type_samples))
                    
                    # Then add more samples randomly up to questions_per_scene
                    remaining = questions_per_scene - len(selected_samples)
                    if remaining > 0 and len(scene_samples) > len(selected_samples):
                        remaining_samples = [s for s in scene_samples if s not in selected_samples]
                        selected_samples.extend(
                            random.sample(remaining_samples, min(remaining, len(remaining_samples)))
                        )
                    
                    dataset.extend(selected_samples)
                
            except FileNotFoundError:
                print(f"Failed to load scene {scene_id}")
                continue
        
        return dataset

    def generate_dataset_object_based(self, total_samples: int = 500, quota: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Generate VQA dataset based on objects with quotas.

        Args:
            total_samples: Total number of samples to generate
            quota: Dictionary mapping object types to minimum quota counts
                  e.g., {"vehicle": 100, "pedestrian": 50}

        Returns:
            List of VQA samples
        """
        dataset = []
        
        # Initialize quotas if not provided
        if quota is None:
            quota = {
                "vehicle": int(total_samples * 0.4),
                "pedestrian": int(total_samples * 0.2),
                "bicycle": int(total_samples * 0.1),
                "traffic_sign": int(total_samples * 0.1),
                "other": int(total_samples * 0.2)
            }
        
        # Initialize quota tracking
        quota_counts = {k: 0 for k in quota}
        
        # Get all scenes
        scene_ids = self.find_scenes()
        random.shuffle(scene_ids)
        
        all_samples = []
        
        # First pass: generate a large pool of samples
        for scene_id in scene_ids:
            try:
                # Load scene
                scene = self.loader.load_scene(scene_id)
                
                # Load objects for this scene
                objects = self.find_scene_objects(scene_id)
                
                if not objects:
                    continue
                
                # Group objects by type
                objects_by_type = defaultdict(list)
                for obj in objects:
                    if obj.cvat_label:
                        obj_type = obj.cvat_label.lower()
                        objects_by_type[obj_type].append(obj)
                
                # Generate samples for each object type with unfilled quota
                for obj_type, objs in objects_by_type.items():
                    # Map specific object types to quota categories
                    quota_category = None
                    for category in quota:
                        if category.lower() in obj_type.lower():
                            quota_category = category
                            break
                    
                    if not quota_category:
                        quota_category = "other"
                    
                    # Skip if quota already filled
                    if quota_counts[quota_category] >= quota[quota_category]:
                        continue
                    
                    # Sample a few objects
                    sampled_objects = random.sample(objs, min(3, len(objs)))
                    
                    for obj in sampled_objects:
                        # Pick a timestamp for this object
                        if not obj.frames:
                            continue
                        
                        timestamp = random.choice(obj.frames)
                        
                        try:
                            frame = self.loader.load_frame(scene_id, timestamp)
                            
                            # Get all objects visible in this frame
                            visible_objects = [o for o in objects if timestamp in o.frames]
                            
                            # Generate samples
                            samples_for_obj = []
                            for generator in self.prompt_generators:
                                samples = generator.generate(scene, visible_objects, frame)
                                # Filter to keep only samples involving the target object
                                # for sample in samples:
                                #     if ("object_id" in sample and sample["object_id"] == obj.id) or \
                                #        ("object_ids" in sample and obj.id in sample["object_ids"]):
                                #         samples_for_obj.append(sample)
                            
                                all_samples.extend(samples)
                            print('all_samples', len(all_samples))
                        except FileNotFoundError:
                            continue
            except FileNotFoundError:
                continue
                
            # Stop if we have enough samples
            if len(all_samples) >= total_samples * 3:  # Generate 3x to allow for selection
                break
        
        # Second pass: select samples to meet quotas
        selected_samples = []
        
        # First, group samples by object type
        samples_by_category = defaultdict(list)
        for sample in all_samples:
            obj_ids = []
            if "object_id" in sample:
                obj_ids = [sample["object_id"]]
            elif "object_ids" in sample:
                obj_ids = sample["object_ids"]
            
            for obj_id in obj_ids:
                # Find object type from metadata
                if "metadata" in sample:
                    metadata = sample["metadata"]
                    obj_label = None
                    
                    # Check different metadata fields for the object label
                    if "target_label" in metadata:
                        obj_label = metadata["target_label"].lower()
                    elif "label" in metadata:
                        obj_label = metadata["label"].lower()
                    elif "object_counts" in metadata:
                        # Scene description - categorize as "other"
                        samples_by_category["other"].append(sample)
                        continue
                    
                    if not obj_label:
                        continue
                    
                    # Map to quota category
                    assigned = False
                    for category in quota:
                        if category.lower() in obj_label:
                            samples_by_category[category].append(sample)
                            assigned = True
                            break
                    
                    if not assigned:
                        samples_by_category["other"].append(sample)
        
        # Select samples to fill quotas
        for category, count in quota.items():
            category_samples = samples_by_category[category]
            random.shuffle(category_samples)
            
            # Select up to quota count
            selected_count = min(count, len(category_samples))
            selected_samples.extend(category_samples[:selected_count])
            quota_counts[category] = selected_count
        
        # Fill remaining slots with any samples
        remaining = total_samples - len(selected_samples)
        if remaining > 0:
            # Pool all unused samples
            unused_samples = [s for s in all_samples if s not in selected_samples]
            random.shuffle(unused_samples)
            
            # Add up to remaining count
            selected_samples.extend(unused_samples[:remaining])
        
        return selected_samples

    def generate_dataset(self, questions_per_scene: int = 5, total_samples: int = 500, 
                         method: str = "combined", quota: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Generate VQA dataset using the specified method.

        Args:
            questions_per_scene: Number of questions per scene for scene-based generation
            total_samples: Total number of samples for object-based generation
            method: Generation method - "scene", "object", or "combined"
            quota: Dictionary mapping object types to minimum quota counts

        Returns:
            List of VQA samples
        """
        if method == "scene":
            return self.generate_dataset_scene_based(questions_per_scene)
        elif method == "object":
            return self.generate_dataset_object_based(total_samples, quota)
        elif method == "combined":
            # Generate half using scene-based approach and half using object-based
            scene_samples = self.generate_dataset_scene_based(questions_per_scene)
            
            # Adjust object quotas to account for scene-based samples
            if quota:
                adjusted_quota = {k: max(0, v // 2) for k, v in quota.items()}
            else:
                adjusted_quota = None
                
            object_samples = self.generate_dataset_object_based(total_samples // 2, adjusted_quota)
            
            # Combine and shuffle
            combined = scene_samples + object_samples
            random.shuffle(combined)
            
            return combined
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'scene', 'object', or 'combined'")
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: Path):
        """Save generated dataset to file."""
        output_path = Path(output_path)
        assert output_path.parent.exists()
        
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
            
        print(f"Saved {len(dataset)} VQA samples to {output_path}")
        
    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the generated dataset and return statistics."""
        stats = {
            "total_samples": len(dataset),
            "question_types": defaultdict(int),
            "scenes": defaultdict(int),
            "object_types": defaultdict(int),
            "objects_per_question": [],
        }
        
        for sample in dataset:
            # Count question types
            q_type = sample.get("question_type", "unknown")
            stats["question_types"][q_type] += 1
            
            # Count scenes
            scene_id = sample.get("scene_id", "unknown")
            stats["scenes"][scene_id] += 1
            
            # Count object types
            if "metadata" in sample:
                metadata = sample["metadata"]
                
                # Check for object label in different metadata fields
                if "target_label" in metadata:
                    label = metadata["target_label"].lower()
                    stats["object_types"][label] += 1
                elif "label" in metadata:
                    label = metadata["label"].lower()
                    stats["object_types"][label] += 1
                elif "object1" in metadata and "label" in metadata["object1"]:
                    label = metadata["object1"]["label"].lower()
                    stats["object_types"][label] += 1
                elif "object_counts" in metadata:
                    for obj_type, count in metadata["object_counts"].items():
                        stats["object_types"][obj_type.lower()] += count
            
            # Count objects per question
            obj_count = 0
            if "object_id" in sample:
                obj_count = 1
            elif "object_ids" in sample:
                obj_count = len(sample["object_ids"])
            
            stats["objects_per_question"].append(obj_count)
        
        # Calculate average and distribution of objects per question
        if stats["objects_per_question"]:
            stats["avg_objects_per_question"] = sum(stats["objects_per_question"]) / len(stats["objects_per_question"])
            stats["max_objects_per_question"] = max(stats["objects_per_question"])
            stats["min_objects_per_question"] = min(stats["objects_per_question"])
        
        # Convert defaultdicts to regular dicts for serialization
        stats["question_types"] = dict(stats["question_types"])
        stats["scenes"] = dict(stats["scenes"])
        stats["object_types"] = dict(stats["object_types"])
        
        return stats


def main():
    """Main function to generate and save a VQA dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VQA dataset from processed data")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--method", type=str, default="combined", choices=["scene", "object", "combined"],
                        help="Generation method")
    parser.add_argument("--questions_per_scene", type=int, default=5, 
                        help="Number of questions per scene for scene-based generation")
    parser.add_argument("--total_samples", type=int, default=500,
                        help="Total number of samples for object-based generation")
    parser.add_argument("--generators", type=str, nargs="+", default=None,
                        help="Specific prompt generators to use (by name)")
    parser.add_argument("--quota_file", type=str, default=None,
                        help="Path to JSON file with object quotas")
    parser.add_argument("--analyze", action="store_true", help="Analyze generated dataset")
    
    args = parser.parse_args()
    
    # Load quotas if provided
    quota = None
    if args.quota_file:
        with open(args.quota_file, "r") as f:
            quota = json.load(f)
    
    # Initialize generator
    generator = VQAGenerator(
        dataset_path=args.dataset_path,
        prompt_generators=args.generators
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(
        questions_per_scene=args.questions_per_scene,
        total_samples=args.total_samples,
        method=args.method,
        quota=quota
    )
    
    output_path = Path(args.dataset_path) / 'vqa.json'
    
    # Save dataset
    generator.save_dataset(dataset, output_path)
    
    # Analyze dataset if requested
    if args.analyze:
        stats = generator.analyze_dataset(dataset)
        stats_path = Path(output_path).with_suffix('.stats.json')
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved dataset statistics to {stats_path}")


if __name__ == "__main__":
    main()