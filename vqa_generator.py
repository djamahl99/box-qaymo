import math
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
        balance_scenes: bool = True,
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
                balance_scenes=balance_scenes,
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
        balance_scenes: bool = False,
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
                    split_name,
                    balance_scenes=balance_scenes,
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

    def _log_balance_stats(
        self, generator_name: str, split_name: str, stats: Dict, output_dir: Path
    ):
        """Log balancing statistics to a file."""
        stats_file = output_dir / f"{generator_name}_{split_name}_balance_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Balance stats logged to {stats_file}")

    def _calculate_balance_metrics(self, answer_counts: Dict[str, int]) -> Dict:
        """Calculate balance metrics for answer distribution."""
        total = sum(answer_counts.values())
        num_choices = len(answer_counts)

        # Standard metrics
        max_count = max(answer_counts.values())
        min_count = min(answer_counts.values())

        # VQA-specific: Check if any answer dominates too much
        dominant_ratio = max_count / total

        # Effective number of answers (entropy-based)
        probs = [count / total for count in answer_counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        effective_answers = math.exp(entropy)  # How many "effective" answer choices

        # Balance score: how close to having all answers equally represented
        balance_score = effective_answers / num_choices  # 1.0 = perfect balance

        return {
            "dominant_ratio": dominant_ratio,
            "effective_answers": effective_answers,
            "balance_score": balance_score,
            "min_count": min_count,
            "max_count": max_count,
            "entropy": entropy,
            "num_choices": num_choices,
        }

    def _should_balance_vqa(self, metrics, num_choices):
        """Simple rules for when to balance VQA multichoice"""

        # Don't balance if already reasonable
        if metrics["balance_score"] > 0.7:  # Pretty balanced already
            return False

        # Always balance if one answer dominates heavily
        if metrics["dominant_ratio"] > 0.6:  # One answer is >60% of responses
            return True

        # Balance if we have very few samples for some answers
        if metrics["min_count"] < 2 and num_choices > 2:
            return True

        return False

    def _balance_samples(
        self,
        samples: List[Tuple],
        target_count: int,
        generator_name: str,
        output_dir: Path,
        split_name: str,
        balance_scenes: bool = False,  # Whether to balance by scene_id
        scene_balance_thresh: float = 0.1,  # Minimum scene representation ratio
    ) -> List[Tuple]:
        """
        Advanced balancing for multi-choice answers with proper error handling.

        Args:
            samples: List of (question, answer) tuples
            target_count: Target number of samples after balancing
            generator_name: Name of the generator for logging
            output_dir: Directory to save balance statistics
            balance_scenes: Whether to apply scene balancing
            scene_balance_thresh: Minimum ratio for scene representation
        """
        if math.isinf(target_count):
            target_count = len(samples)

        # First apply scene balancing if requested
        before_scene_balance = len(samples)
        if balance_scenes:
            samples = self._balance_by_scenes(
                samples, generator_name, scene_balance_thresh
            )
            print(
                f"After scene balancing: {len(samples)} samples remaining out of {before_scene_balance}"
            )

        # Then apply answer balancing as before
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
            if isinstance(answer, MultipleChoiceAnswer):
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

            num_choices = metrics["num_choices"]

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
            print(f"  effective_answers: {metrics['effective_answers']:.3f}")
            print(f"  balance_score: {metrics['balance_score']:.3f}")
            print(f"  entropy: {metrics['entropy']:.3f}")

            # Apply balancing logic based on answer type
            if is_multichoice and len(answer_counts) > 1:
                # Check if balancing is needed for multi-choice
                if self._should_balance_vqa(metrics, num_choices):
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
        }

        self._log_balance_stats(generator_name, split_name, summary_stats, output_dir)

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

    def _calculate_sqrt_targets(self, answer_counts, target_total):
        """Square root balancing - compromise between uniform and original distribution"""
        sqrt_counts = {answer: math.sqrt(count) if count > 0 else 0 for answer, count in answer_counts.items()}
        total_sqrt = sum(sqrt_counts.values())
        
        print('total_sqrt', total_sqrt)
        
        targets = {}
        for answer, sqrt_count in sqrt_counts.items():
            if total_sqrt == 0:
                print(f'total_sqrt={total_sqrt}, answer_counts[answer]={answer_counts[answer]}')
                targets[answer] = 0
            else:
                print(f'{target_total} * {sqrt_count} / {total_sqrt}')
                targets[answer] = max(1, int(target_total * sqrt_count / total_sqrt))
        
        return targets


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

        # Ensure minimum representation (at least 2-3 samples per valid answer)
        min_per_answer = max(2, target_count // (len(answer_counts) * 4))  # At least 2, but not too high
        
        # Calculate sqrt-based targets
        sqrt_targets = self._calculate_sqrt_targets(answer_counts, target_count)
        
        # Apply minimum guarantees
        final_targets = {}
        for answer in answer_counts:
            final_targets[answer] = max(min_per_answer, sqrt_targets[answer])
        
        # Scale down if total exceeds target
        total_planned = sum(final_targets.values())
        if total_planned > target_count:
            scale_factor = target_count / total_planned
            final_targets = {answer: max(1, int(count * scale_factor)) 
                            for answer, count in final_targets.items()}

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
            target_for_answer = final_targets[answer]

            # Don't exceed available samples
            actual_count = min(target_for_answer, len(available_indices))

            # Select samples for this answer
            selected_for_answer = random.sample(available_indices, actual_count)
            selected_indices.extend(selected_for_answer)

        return selected_indices

    def _balance_by_scenes(
        self,
        samples: List[Tuple],
        generator_name: str,
        scene_balance_thresh: float = 0.1,
    ) -> List[Tuple]:
        """
        Balance samples to ensure reasonable scene distribution.

        Args:
            samples: List of (question, answer) tuples
            generator_name: Name of generator for logging
            scene_balance_thresh: Minimum ratio for scene representation
        """
        print(f"\nApplying scene balancing for {generator_name}")

        # Group samples by scene_id
        scene_to_samples = {}
        samples_without_scene = []

        for sample in samples:
            question, answer = sample
            scene_id = getattr(question, "scene_id", None)

            if scene_id is not None:
                scene_to_samples.setdefault(scene_id, [])
                scene_to_samples[scene_id].append(sample)
            else:
                samples_without_scene.append(sample)

        if not scene_to_samples:
            print(f"  No scene_id found in samples for {generator_name}")
            return samples

        # Calculate scene distribution
        scene_counts = {
            scene_id: len(samples_list)
            for scene_id, samples_list in scene_to_samples.items()
        }
        total_samples = sum(scene_counts.values())
        num_scenes = len(scene_counts)

        print(f"  Found {num_scenes} scenes with {total_samples} samples")
        print(
            f"  Scene distribution: {dict(sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)[:10])}..."
        )

        # Calculate balance metrics for scenes
        scene_metrics = self._calculate_balance_metrics(scene_counts)
        print(f"  Scenes effective: {scene_metrics['effective_answers']:.3f}")
        print(f"  Scene dominant ratio: {scene_metrics['dominant_ratio']:.3f}")
        print(f"  Scene balance_score: {scene_metrics['balance_score']:.3f}")
        print(f"  Scene entropy: {scene_metrics['entropy']:.3f}")

        # Check if balancing is needed
        if (
            scene_metrics["dominant_ratio"] > 0.5
            or scene_metrics["balance_score"] < scene_balance_thresh
        ):
            print(f"  Scene imbalance detected - applying balancing")
            balanced_samples = self._apply_scene_balancing(
                scene_to_samples, scene_counts
            )
        else:
            print(f"  Scenes already reasonably balanced")
            balanced_samples = [
                sample
                for samples_list in scene_to_samples.values()
                for sample in samples_list
            ]

        # Add back samples without scene_id
        balanced_samples.extend(samples_without_scene)

        # Log final scene distribution
        final_scene_counts = {}
        for sample in balanced_samples:
            scene_id = getattr(sample[0], "scene_id", "no_scene")
            final_scene_counts[scene_id] = final_scene_counts.get(scene_id, 0) + 1

        print(
            f"  Final scene distribution: {dict(sorted(final_scene_counts.items(), key=lambda x: x[1], reverse=True)[:10])}..."
        )

        return balanced_samples

    def _apply_scene_balancing(
        self, scene_to_samples: Dict[str, List[Tuple]], scene_counts: Dict[str, int]
    ) -> List[Tuple]:
        """
        Apply scene balancing by capping over-represented scenes.
        """
        total_samples = sum(scene_counts.values())
        num_scenes = len(scene_counts)

        # Calculate target samples per scene (with some flexibility for variation)
        # We don't want perfect uniformity, but prevent extreme dominance
        mean_samples_per_scene = total_samples / num_scenes
        max_samples_per_scene = int(mean_samples_per_scene * 2)  # Allow 2x the mean
        min_samples_per_scene = max(
            1, int(mean_samples_per_scene * 0.5)
        )  # At least 50% of mean

        balanced_samples = []

        for scene_id, samples_list in scene_to_samples.items():
            current_count = len(samples_list)

            if current_count > max_samples_per_scene:
                # Too many samples from this scene - randomly sample
                selected_samples = random.sample(samples_list, max_samples_per_scene)
                print(
                    f"    Capped scene {scene_id}: {current_count} -> {max_samples_per_scene}"
                )
            else:
                # Keep all samples from this scene
                selected_samples = samples_list

            balanced_samples.extend(selected_samples)

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
    parser.add_argument("--save_prefix", type=str, required=False, default="")
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
        "--balance_scenes", action="store_true", help="Balance the scenes"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Save samples in batches of this size.",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = VQAGenerator(
        dataset_path=args.dataset_path, prompt_generators=args.generators
    )

    print("args.total_samples", args.total_samples)
    generator.generate_dataset_per_generator(
        total_samples=args.total_samples,
        visualise=args.visualise,
        save_prefix=args.save_prefix,
        batch_size=args.batch_size,
        balance_scenes=args.balance_scenes,
    )
    print("Dataset generation completed. Files saved separately for each generator.")


if __name__ == "__main__":
    main()
