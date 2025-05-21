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
from waymovqa.prompt_generators import (
    get_all_prompt_generators,
    get_prompt_generator,
    BasePromptGenerator,
)
from waymovqa.models import MODEL_REGISTRY
from waymovqa.models.base import BaseModel

from waymovqa.data.vqa_dataset import VQADataset
from tqdm import tqdm


class VQAGenerator:
    """Main class for generating VQA samples."""

    prompt_generators: List[BasePromptGenerator]
    scene_prompt_generators: List[BasePromptGenerator]
    object_prompt_generators: List[BasePromptGenerator]
    frame_prompt_generators: List[BasePromptGenerator]

    def __init__(
        self,
        dataset_path: Path,
        prompt_generators: Optional[Union[List[str], List[BasePromptGenerator]]] = None,
        model: BaseModel = None,  # TODO - infer prompt generators from model
    ):
        """
        Initialize VQA generator.

        Args:
            prompt_generators: Optional list of prompt generator names or instances.
                               If None, all registered generators will be used.
        """
        self.dataset_path = Path(dataset_path)
        self.loader = WaymoDatasetLoader(self.dataset_path)
        self.dataset = VQADataset(tag="ground_truth")

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

    def generate_dataset(self, total_samples: int = 500, balance_answers: bool = True):
        """Generate VQA dataset."""
        all_samples = []

        # Load a subset of scenes
        scene_ids = self.loader.get_scene_ids()
        random.shuffle(scene_ids)

        # Process each scene
        for scene_id in scene_ids:
            # Load all frames for this scene
            frames = self._load_all_frames(scene_id)

            # Apply each generator to all frames
            for generator in self.prompt_generators:
                samples = generator.generate(frames)
                all_samples.extend(samples)

        # Balance dataset if needed
        if balance_answers:
            final_samples = self._balance_samples(all_samples, total_samples)
        else:
            final_samples = random.sample(
                all_samples, min(total_samples, len(all_samples))
            )

        # Add to dataset
        for sample in final_samples:
            self.dataset.add_sample(*sample)

    def _balance_samples(self, samples: List[Tuple], target_count: int) -> List[Tuple]:
        """
        Balance dataset to have more uniform distribution of answers.
        Returns a subset of samples with a more balanced distribution.
        """
        # Group samples by the prompt generator
        generator_samples = {}
        for sample_idx, (question, answer) in enumerate(samples):
            if answer.get_answer_text() is None:  # don't balance grounding 2d...
                continue

            generator_samples.setdefault(question.generator_name, [])
            generator_samples[question.generator_name].append(sample_idx)

        # Final indices after balancing
        final_indices = []

        for generator_name, generator_sample_indices in generator_samples.items():
            print("generator_name", generator_name)
            # Count frequency of each answer
            answer_counts = {}
            for sample_idx in generator_sample_indices:
                answer = samples[sample_idx][1].get_answer_text()
                answer_counts[answer] = answer_counts.get(answer, 0) + 1

            print(f"Answer distribution before balancing: {answer_counts}")

            # Calculate sample weights (inverse of frequency)
            weights = []
            for _, answer in samples:
                frequency = answer_counts[answer.answer]
                weight = 1.0 / frequency if frequency > 0 else 0
                weights.append(weight)

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Weighted random sampling
            balanced_indices = random.choices(
                range(len(samples)), weights=weights, k=min(target_count, len(samples))
            )

            # Print statistics after balancing
            balanced_answers = [
                samples[i][1].get_answer_text() for i in balanced_indices
            ]
            balanced_counts = {}
            for answer in balanced_answers:
                balanced_counts[answer] = balanced_counts.get(answer, 0) + 1

            final_indices.extend(balanced_indices)

        balanced_samples = [samples[i] for i in final_indices]

        return balanced_samples

    def _collect_all_object_contexts(self) -> List[Tuple[str, int]]:
        """Collect all valid (scene_id, timestamp) combinations."""
        contexts = []
        scene_ids = self.loader.get_scene_ids()

        for scene_id in scene_ids:
            timestamps = self.loader.get_frame_timestamps(scene_id)
            for timestamp in timestamps:
                contexts.append((scene_id, timestamp))

        # Shuffle to avoid bias
        random.shuffle(contexts)
        return contexts

    def _collect_all_frame_contexts(self) -> List[Tuple[str, int]]:
        """Collect all valid (scene_id, timestamp) combinations."""
        contexts = []
        scene_ids = self.loader.get_scene_ids()

        for scene_id in scene_ids:
            timestamps = self.loader.get_frame_timestamps(scene_id)
            for timestamp in timestamps:
                contexts.append((scene_id, timestamp))

        # Shuffle to avoid bias
        random.shuffle(contexts)
        return contexts

    def _collect_all_scene_contexts(self) -> List[str]:
        """Collect all valid (scene_id, timestamp) combinations."""
        scene_ids = self.loader.get_scene_ids()

        # Shuffle to avoid bias
        random.shuffle(scene_ids)
        return scene_ids


def main():
    """Main function to generate and save a VQA dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VQA dataset from processed data"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to processed waymo dataset",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save dataset"
    )
    parser.add_argument(
        "--model", type=str, default=None, required=False, help="Model name"
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=500,
        help="Total number of samples for object-based generation",
    )
    parser.add_argument(
        "--generators",
        type=str,
        nargs="+",
        default=None,
        help="Specific prompt generators to use (by name)",
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze generated dataset"
    )

    args = parser.parse_args()

    if args.model is not None:
        model = MODEL_REGISTRY[args.model.lower()]
    else:
        model = None

    # Initialize generator
    generator = VQAGenerator(
        dataset_path=args.dataset_path, prompt_generators=args.generators, model=model
    )

    # Generate dataset
    generator.generate_dataset(total_samples=args.total_samples)

    output_path = Path(args.save_path)

    # Save dataset
    generator.dataset.save_dataset(str(output_path))


if __name__ == "__main__":
    main()
