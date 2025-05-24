import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
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

    def _get_visualization_filename(
        self, question_obj, answer_obj, generator: BasePromptGenerator
    ) -> str:
        """Generate a descriptive filename for visualization."""

        # descriptive name based on question properties
        parts = []

        # Add scene and timestamp info if available
        # if hasattr(question_obj, "scene_id"):
        #     parts.append(f"scene_{question_obj.scene_id}")
        # if hasattr(question_obj, "timestamp"):
        #     parts.append(f"t_{question_obj.timestamp}")
        # if hasattr(question_obj, "camera_name"):
        # parts.append(f"cam_{question_obj.camera_name}")
        if hasattr(question_obj, "question_name"):
            parts.append(f"{question_obj.question_name}")
        if hasattr(question_obj, "question_id"):
            parts.append(f"{question_obj.question_id}")

        base_name = "_".join(parts)

        # Clean filename and add generator prefix
        clean_name = "".join(c for c in base_name if c.isalnum() or c in "._-")
        return f"{generator.__class__.__name__}_{clean_name}"

    def _visualise_sample(
        self, generator: BasePromptGenerator, sample, frames: List[FrameInfo]
    ):
        save_dir = self.dataset_path / "visualisation"
        save_dir.mkdir(exist_ok=True)

        generator_save_dir = save_dir / f"{generator.__class__.__name__}"
        generator_save_dir.mkdir(exist_ok=True)

        question_obj, answer_obj = sample

        # Use the new filename generation method
        filename = self._get_visualization_filename(question_obj, answer_obj, generator)
        question_save_path = generator_save_dir / f"{filename}.png"

        # Handle filename conflicts by adding counter
        counter = 1
        while question_save_path.exists():
            question_save_path = generator_save_dir / f"{filename}_{counter}.png"
            counter += 1

        generator.visualise_sample(question_obj, answer_obj, question_save_path, frames)

    def generate_dataset_per_generator(
        self,
        total_samples: int = 500,
        balance_answers: bool = True,
        visualise: bool = False,
        save_prefix: str = "",
        batch_size: Optional[int] = None,
        balance_thresh: float = 0.15,
        dominant_thresh: float = 0.8,
        gini_thresh: float = 0.6,
    ):
        """
        Generate VQA dataset by processing each generator separately.
        This avoids creating large JSON files by saving each generator's output individually.
        """
        output_dir = Path(self.dataset_path) / "generated_vqa_samples"
        output_dir.mkdir(exist_ok=True)

        # Process each generator separately
        for generator in self.prompt_generators:
            print(f"\n{'='*50}")
            print(f"Processing generator: {generator.__class__.__name__}")
            print(f"{'='*50}")

            self._process_single_generator(
                generator=generator,
                total_samples=total_samples,
                balance_answers=balance_answers,
                visualise=visualise,
                output_dir=output_dir,
                save_prefix=save_prefix,
                batch_size=batch_size,
                balance_thresh=balance_thresh,
                dominant_thresh=dominant_thresh,
                gini_thresh=gini_thresh,
            )

    def _process_single_generator(
        self,
        generator: BasePromptGenerator,
        total_samples: int,
        balance_answers: bool,
        visualise: bool,
        output_dir: Path,
        save_prefix: str,
        batch_size: Optional[int] = None,
        balance_thresh: float = 0.15,
        dominant_thresh: float = 0.8,
        gini_thresh: float = 0.6,
    ):
        """Process a single generator and save its output."""
        generator_name = generator.__class__.__name__

        for split_name, split_path in self.split_paths.items():
            print(f"\nProcessing {split_name} split for {generator_name}")

            assert split_path.exists(), f"Split path does not exist: {split_path}"
            with open(split_path, "r") as f:
                lines = f.readlines()

            scene_ids = [x.strip() for x in lines]
            random.shuffle(scene_ids)

            all_samples = []

            # Process each scene for this generator
            for scene_id in tqdm(
                scene_ids, desc=f"Generating {split_name} for {generator_name}"
            ):
                # Load all frames for this scene
                frames = self._load_all_frames(scene_id)

                # Apply the current generator to all frames
                samples = generator.generate(frames)

                if len(samples) == 0:
                    print(f"{generator_name} generated no samples for {scene_id}")
                    continue

                if visualise and len(samples) > 0:
                    for vis_idx in np.random.choice(
                        len(samples), min(5, len(samples)), replace=False
                    ):
                        vis_sample = samples[vis_idx]
                        self._visualise_sample(generator, vis_sample, frames)

                all_samples.extend(samples)

                if len(all_samples) >= total_samples * 1.5:
                    print(f"Reached sample limit for {generator_name} on {split_name}")
                    break

            if not all_samples:
                print(f"No samples generated for {generator_name} on {split_name}")
                continue

            # Balance dataset if needed
            if balance_answers:
                final_samples = self._balance_samples(
                    all_samples,
                    total_samples,
                    generator_name,
                    output_dir,
                    balance_thresh=balance_thresh,
                    dominant_thresh=dominant_thresh,
                    gini_thresh=gini_thresh,
                )
            else:
                final_samples = random.sample(
                    all_samples, min(total_samples, len(all_samples))
                )

            print(
                f"Final sample count for {generator_name} {split_name}: {len(final_samples)}"
            )

            # Save this generator's samples
            if batch_size and len(final_samples) > batch_size:
                self._save_generator_in_batches(
                    final_samples,
                    generator_name,
                    split_name,
                    output_dir,
                    save_prefix,
                    batch_size,
                )
            else:
                self._save_generator_single_file(
                    final_samples, generator_name, split_name, output_dir, save_prefix
                )

    def _save_generator_single_file(
        self,
        samples: List[Tuple],
        generator_name: str,
        split_name: str,
        output_dir: Path,
        save_prefix: str,
    ):
        """Save all samples from a generator to a single file."""
        # Create dataset for this generator
        dataset = VQADataset(tag=f"{split_name}_{generator_name}_ground_truth")

        for sample in samples:
            dataset.add_sample(*sample)

        # Save to file
        filename_parts = [
            part for part in [save_prefix, generator_name, split_name] if part
        ]
        filename = "_".join(filename_parts) + ".json"
        save_path = output_dir / filename

        print(f"Saving {len(samples)} samples to {save_path}")
        dataset.save_dataset(str(save_path))

    def _save_generator_in_batches(
        self,
        samples: List[Tuple],
        generator_name: str,
        split_name: str,
        output_dir: Path,
        save_prefix: str,
        batch_size: int,
    ):
        """Save generator samples in multiple batch files."""
        num_batches = (len(samples) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            # Create dataset for this batch
            dataset = VQADataset(
                tag=f"{split_name}_{generator_name}_batch_{batch_idx}_ground_truth"
            )

            for sample in batch_samples:
                dataset.add_sample(*sample)

            # Save batch to file
            filename_parts = [
                part
                for part in [
                    save_prefix,
                    generator_name,
                    split_name,
                    f"batch_{batch_idx}",
                ]
                if part
            ]
            filename = "_".join(filename_parts) + ".json"
            save_path = output_dir / filename

            print(
                f"Saving batch {batch_idx + 1}/{num_batches} ({len(batch_samples)} samples) to {save_path}"
            )
            dataset.save_dataset(str(save_path))

    def _load_all_frames(self, scene_id) -> List[FrameInfo]:
        """Load all frames for a scene_id"""
        timestamps = self.loader.get_frame_timestamps(scene_id)
        frames = []
        for timestamp in timestamps:
            frames.append(self.loader.load_frame(scene_id, timestamp))

        return frames

    def _is_multichoice_answer(self, answer) -> bool:
        """Check if an answer is a multi-choice answer type."""
        # Add your specific logic here based on your answer types
        # This is a placeholder - adjust based on your actual answer classes
        return (
            hasattr(answer, "choices")
            or hasattr(answer, "options")
            or answer.__class__.__name__.lower().find("choice") != -1
            or answer.__class__.__name__.lower().find("multichoice") != -1
        )

    def _log_balance_stats(self, generator_name: str, stats: Dict, output_dir: Path):
        """Log balancing statistics to a file."""
        stats_file = output_dir / f"{generator_name}_balance_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Balance stats logged to {stats_file}")

    def _calculate_balance_metrics(self, answer_counts: Dict[str, int]) -> Dict:
        """Calculate balance metrics for answer distribution."""
        total_answers = sum(answer_counts.values())
        num_choices = len(answer_counts)

        if num_choices == 0:
            return {"balance_ratio": 0, "gini_coefficient": 1.0, "dominant_ratio": 1.0}

        expected_count = total_answers / num_choices

        # Calculate Gini coefficient for inequality measurement
        sorted_counts = sorted(answer_counts.values())
        n = len(sorted_counts)
        gini = (2 * sum((i + 1) * count for i, count in enumerate(sorted_counts))) / (
            n * total_answers
        ) - (n + 1) / n

        # Calculate dominant answer ratio
        max_count = max(answer_counts.values())
        dominant_ratio = max_count / total_answers

        # Balance ratio (how close to uniform distribution)
        min_count = min(answer_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0

        return {
            "balance_ratio": balance_ratio,
            "gini_coefficient": gini,
            "dominant_ratio": dominant_ratio,
            "expected_count_per_choice": expected_count,
            "total_answers": total_answers,
            "num_choices": num_choices,
        }

    def _balance_samples(
        self,
        samples: List[Tuple],
        target_count: int,
        generator_name: str,
        output_dir: Path,
        balance_thresh: float = 0.15,  # Minimum ratio of least common to most common
        dominant_thresh: float = 0.8,  # Maximum ratio for dominant answer
        gini_thresh: float = 0.6,  # Maximum Gini coefficient
    ) -> List[Tuple]:
        """
        Advanced balancing for multi-choice answers with proper error handling.

        Args:
            samples: List of (question, answer) tuples
            target_count: Target number of samples after balancing
            generator_name: Name of the generator for logging
            output_dir: Directory to save balance statistics
            balance_thresh: Minimum balance ratio to accept
            dominant_thresh: Maximum dominant answer ratio to accept
            gini_thresh: Maximum Gini coefficient to accept
        """
        # Group samples by question type
        question_groups = {}
        multichoice_groups = {}

        for sample_idx, (question, answer) in enumerate(samples):
            if answer.get_answer_text() is None:
                continue

            question_name = question.generator_name
            if (
                hasattr(question, "question_name")
                and question.question_name is not None
            ):
                question_name += f"_{question.question_name}"

            question_groups.setdefault(question_name, [])
            question_groups[question_name].append(sample_idx)

            # Separate multi-choice questions
            if isinstance(MultipleChoiceAnswer, answer):
                multichoice_groups.setdefault(question_name, [])
                multichoice_groups[question_name].append(sample_idx)

        final_indices = []
        balance_stats = {}
        errors = []
        warnings = []

        print(f"\nBalancing samples for generator: {generator_name}")
        print(f"Question groups: {list(question_groups.keys())}")
        print(f"Multi-choice groups: {list(multichoice_groups.keys())}")

        for question_name, sample_indices in question_groups.items():
            is_multichoice = question_name in multichoice_groups

            # Count answer frequencies
            answer_counts = {}
            for sample_idx in sample_indices:
                answer_text = samples[sample_idx][1].get_answer_text()
                answer_counts[answer_text] = answer_counts.get(answer_text, 0) + 1

            # Calculate balance metrics
            metrics = self._calculate_balance_metrics(answer_counts)

            question_stats = {
                "question_name": question_name,
                "is_multichoice": is_multichoice,
                "answer_distribution": answer_counts,
                "metrics": metrics,
                "total_samples": len(sample_indices),
            }

            print(
                f"\n{question_name} ({'Multi-choice' if is_multichoice else 'Open-ended'}):"
            )
            print(f"  Answer distribution: {answer_counts}")
            print(f"  Balance ratio: {metrics['balance_ratio']:.3f}")
            print(f"  Gini coefficient: {metrics['gini_coefficient']:.3f}")
            print(f"  Dominant ratio: {metrics['dominant_ratio']:.3f}")

            # Apply balancing logic based on answer type
            if is_multichoice and len(answer_counts) > 1:
                # Check if balancing is needed for multi-choice
                if (
                    metrics["balance_ratio"] < balance_thresh
                    or metrics["dominant_ratio"] > dominant_thresh
                    or metrics["gini_coefficient"] > gini_thresh
                ):

                    if metrics["dominant_ratio"] > 0.95:
                        # Extremely imbalanced - log error but continue
                        error_msg = f"CRITICAL: {question_name} is extremely imbalanced (dominant ratio: {metrics['dominant_ratio']:.3f})"
                        errors.append(error_msg)
                        print(f"  ERROR: {error_msg}")
                        question_stats["status"] = "error_extreme_imbalance"
                        # Still include samples but log the issue
                        selected_indices = random.sample(
                            sample_indices, min(target_count, len(sample_indices))
                        )
                    else:
                        # Apply balancing by capping dominant answers
                        selected_indices = self._balance_multichoice_answers(
                            sample_indices,
                            samples,
                            answer_counts,
                            target_count,
                            metrics,
                        )
                        question_stats["status"] = "balanced"
                        print(
                            f"  Applied balancing: {len(selected_indices)} samples selected"
                        )
                else:
                    # Already balanced
                    selected_indices = random.sample(
                        sample_indices, min(target_count, len(sample_indices))
                    )
                    question_stats["status"] = "already_balanced"
                    print(
                        f"  Already balanced: {len(selected_indices)} samples selected"
                    )
            else:
                # Non-multichoice or single answer - no balancing needed
                selected_indices = random.sample(
                    sample_indices, min(target_count, len(sample_indices))
                )
                question_stats["status"] = "no_balancing_needed"
                print(
                    f"  No balancing applied: {len(selected_indices)} samples selected"
                )

            # Calculate final distribution
            final_answers = [samples[i][1].get_answer_text() for i in selected_indices]
            final_counts = {}
            for answer in final_answers:
                final_counts[answer] = final_counts.get(answer, 0) + 1

            question_stats["final_distribution"] = final_counts
            question_stats["final_metrics"] = self._calculate_balance_metrics(
                final_counts
            )

            print(f"  Final distribution: {final_counts}")

            balance_stats[question_name] = question_stats
            final_indices.extend(selected_indices)

        # Log comprehensive statistics
        summary_stats = {
            "generator_name": generator_name,
            "total_input_samples": len(samples),
            "total_output_samples": len(final_indices),
            "question_groups": balance_stats,
            "errors": errors,
            "warnings": warnings,
            "thresholds": {
                "balance_thresh": balance_thresh,
                "dominant_thresh": dominant_thresh,
                "gini_thresh": gini_thresh,
            },
        }

        self._log_balance_stats(generator_name, summary_stats, output_dir)

        # Print summary
        if errors:
            print(f"\n⚠️  {len(errors)} ERRORS found in {generator_name}:")
            for error in errors:
                print(f"    {error}")

        if warnings:
            print(f"\n⚠️  {len(warnings)} WARNINGS found in {generator_name}:")
            for warning in warnings:
                print(f"    {warning}")

        # Remove duplicates and return
        final_indices = list(set(final_indices))
        balanced_samples = [samples[i] for i in final_indices]

        print(f"\nFinal result for {generator_name}: {len(balanced_samples)} samples")
        return balanced_samples

    def _balance_multichoice_answers(
        self,
        sample_indices: List[int],
        samples: List[Tuple],
        answer_counts: Dict[str, int],
        target_count: int,
        metrics: Dict,
    ) -> List[int]:
        """
        Balance multi-choice answers by capping dominant answers and ensuring minimum representation.
        """
        total_samples = len(sample_indices)
        num_choices = len(answer_counts)

        # Calculate target count per choice (with some flexibility)
        base_count_per_choice = target_count // num_choices
        remainder = target_count % num_choices

        # Group samples by answer
        answer_to_indices = {}
        for idx in sample_indices:
            answer_text = samples[idx][1].get_answer_text()
            answer_to_indices.setdefault(answer_text, [])
            answer_to_indices[answer_text].append(idx)

        selected_indices = []

        # Sort answers by frequency (ascending) to handle rare answers first
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1])

        for i, (answer, count) in enumerate(sorted_answers):
            available_indices = answer_to_indices[answer]

            # Calculate target for this answer
            target_for_answer = base_count_per_choice
            if i < remainder:  # Distribute remainder among first few answers
                target_for_answer += 1

            # Don't exceed available samples
            actual_count = min(target_for_answer, len(available_indices))

            # Select samples for this answer
            selected_for_answer = random.sample(available_indices, actual_count)
            selected_indices.extend(selected_for_answer)

        return selected_indices


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
    parser.add_argument("--save_prefix", type=str, required=False, default="")
    parser.add_argument(
        "--model", type=str, default=None, required=False, help="Model name"
    )
    parser.add_argument(
        "--total_samples",
        type=float,
        default=float("inf"),
        help="Total number of samples for generation",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Save samples in batches of this size.",
    )
    parser.add_argument(
        "--balance_thresh",
        type=float,
        default=0.15,
        help="Minimum balance ratio for multi-choice answers",
    )
    parser.add_argument(
        "--dominant_thresh",
        type=float,
        default=0.8,
        help="Maximum dominant answer ratio",
    )
    parser.add_argument(
        "--gini_thresh",
        type=float,
        default=0.6,
        help="Maximum Gini coefficient for inequality",
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

    generator.generate_dataset_per_generator(
        total_samples=args.total_samples,
        visualise=args.visualise,
        save_prefix=args.save_prefix,
        batch_size=args.batch_size,
        balance_thresh=args.balance_thresh,
        dominant_thresh=args.dominant_thresh,
        gini_thresh=args.gini_thresh,
    )

if __name__ == "__main__":
    main()
