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

        self.datasets = {
            "training": VQADataset(tag="training_ground_truth"),
            "validation": VQADataset(tag="validation_ground_truth"),
        }

        self.split_paths = {
            "training": self.dataset_path / "splits/train.txt",
            "validation": self.dataset_path / "splits/val.txt",
        }

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

    def _visualise_sample(
        self, generator: BasePromptGenerator, sample, frames: List[FrameInfo]
    ):
        save_dir = self.dataset_path / "visualisation"
        save_dir.mkdir(exist_ok=True)

        generator_save_dir = save_dir / f"{generator.__class__.__name__}"
        generator_save_dir.mkdir(exist_ok=True)

        question_save_path = (
            generator_save_dir / f"{hash(sample[0])}.png"
        )  # hash the question as the filename

        question_obj, answer_obj = sample
        generator.visualise_sample(question_obj, answer_obj, question_save_path, frames)

    def generate_dataset(
        self,
        total_samples: int = 500,
        balance_answers: bool = True,
        visualise: bool = False,
    ):
        """Generate VQA dataset."""
        all_samples = []

        for split_name, split_path in self.split_paths.items():
            assert split_path.exists(), "Split path does not exist!"
            with open(split_path, "r") as f:
                lines = f.readlines()

            scene_ids = [x.strip() for x in lines]
            random.shuffle(scene_ids)

            # Process each scene
            for scene_id in tqdm(scene_ids, desc=f"Generating {split_name} dataset"):
                # Load all frames for this scene
                frames = self._load_all_frames(scene_id)

                # Apply each generator to all frames
                for generator in self.prompt_generators:
                    samples = generator.generate(frames)

                    if len(samples) == 0:
                        print(
                            f"{generator.__class__.__name__} generated no samples for {scene_id}"
                        )

                    if visualise and len(samples) > 0:
                        for vis_idx in np.random.choice(
                            len(samples), min(5, len(samples)), replace=False
                        ):
                            vis_sample = samples[vis_idx]
                            self._visualise_sample(generator, vis_sample, frames)

                    all_samples.extend(samples)

                if len(all_samples) >= total_samples * 1.5:
                    break

            # Balance dataset if needed
            if balance_answers:
                final_samples = self._balance_samples(all_samples, total_samples)
            else:
                final_samples = random.sample(
                    all_samples, min(total_samples, len(all_samples))
                )

            # Add to dataset
            for sample in final_samples:
                self.datasets[split_name].add_sample(*sample)

    def _load_all_frames(self, scene_id) -> List[FrameInfo]:
        """Load all frames for a scene_id"""
        timestamps = self.loader.get_frame_timestamps(scene_id)
        frames = []
        for timestamp in timestamps:
            frames.append(self.loader.load_frame(scene_id, timestamp))

        return frames

    def _balance_samples(
        self, samples: List[Tuple], target_count: int, class_balance_thresh: float = 0.5
    ) -> List[Tuple]:
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

        print("generator_samples.keys()", generator_samples.keys())

        for generator_name, generator_sample_indices in generator_samples.items():
            # Count frequency of each answer
            answer_counts = {}
            total_answers = 0
            for sample_idx in generator_sample_indices:
                answer = samples[sample_idx][1].get_answer_text()
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
                total_answers += 1

            print(f"{generator_name}")
            print(f"Answer distribution before balancing: {answer_counts}")

            # Check if we need to prune
            for answer, answer_count in answer_counts.items():
                if (
                    answer_count
                    < class_balance_thresh * (1.0 / len(answer_counts)) * total_answers
                ):
                    print(
                        "WARNING: Maybe we should prune this answer:", answer
                    )  # TODO: probably no pruning since rare questions are inherent to autonomous driving

            # Calculate sample weights (inverse of frequency)
            weights = []
            for sample_idx in generator_sample_indices:
                answer = samples[sample_idx][1]
                frequency = answer_counts[answer.get_answer_text()]
                weight = 1.0 / frequency if frequency > 0 else 0
                weights.append(weight)

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Weighted random sampling
            balanced_indices = random.choices(
                generator_sample_indices,
                weights=weights,
                k=min(target_count, len(generator_sample_indices)),
            )

            # Print statistics after balancing
            balanced_answers = [
                samples[i][1].get_answer_text() for i in balanced_indices
            ]
            balanced_counts = {}
            for answer in balanced_answers:
                balanced_counts[answer] = balanced_counts.get(answer, 0) + 1

            print(f"Answer distribution AFTER balancing: {answer_counts}")

            final_indices.extend(balanced_indices)

        # Remove duplicates
        final_indices = list(set(final_indices))

        balanced_samples = [samples[i] for i in final_indices]

        return balanced_samples


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
    # parser.add_argument(
    #     "--save_path", type=str, required=True, help="Path to save dataset"
    # )
    parser.add_argument(
        "--model", type=str, default=None, required=False, help="Model name"
    )
    parser.add_argument(
        "--total_samples",
        type=float,
        default=float("inf"),
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
        "--visualise", action="store_true", help="Visualise generated dataset"
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

    print("args.total_samples", args.total_samples)

    # Generate dataset
    generator.generate_dataset(
        total_samples=args.total_samples, visualise=args.visualise
    )

    # output_path = Path(args.save_path)
    output_dir = Path(args.dataset_path) / "generated_vqa_samples"
    # Save dataset
    for split_name, dataset in generator.datasets.items():
        split_save_path = output_dir / f"{split_name}.json"

        print(f"Saving {split_name} to {split_save_path}")

        dataset.save_dataset(str(split_save_path))


if __name__ == "__main__":
    main()
