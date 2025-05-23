import enum
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from waymovqa.questions.multi_image import MultipleImageQuestion
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import (
    OBJECT_SPEED_CATS,
    WAYMO_TYPE_MAPPING,
    HeadingType,
    ObjectInfo,
)
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator
from waymovqa.prompt_generators.templates import *
from waymovqa.primitives import WAYMO_LABEL_TYPES

from functools import partial


@register_prompt_generator
class ObjectBinaryPromptGenerator(BasePromptGenerator):
    """Generates binary (yes/no) questions about object presence."""

    def __init__(self) -> None:
        super().__init__()
        self.negative_sample_ratio = 0.5  # 30% negative samples

        self.prompt_functions = [
            self._object_facing_type,
            self._object_movement_direction_type,
        ]

    def _object_movement_direction_type(self, camera, frame):
        """Generates binary questions about object type, movement status (moving or static) and movement direction."""
        direction_counts = defaultdict(int)

        for obj in frame.objects:
            if obj.type in ["TYPE_UNKNOWN", "TYPE_SIGN"]:
                continue

            if not obj.is_visible_on_camera(frame, camera):
                continue

            movement_dir_txt = obj.get_camera_movement_direction(camera, frame)
            if not movement_dir_txt:
                continue

            obj_type = obj.type
            if obj_type is None:
                continue

            movement_status = obj.get_speed_category()
            key = (obj_type, movement_dir_txt, movement_status)
            direction_counts[key] += 1

        questions = []
        answers = []

        # Add positive examples (where objects exist)
        for (
            obj_type,
            movement_dir_txt,
            movement_status,
        ), count in direction_counts.items():
            answer = "yes" if count > 0 else "no"
            questions.append(
                f"Are there any {movement_status} {WAYMO_TYPE_MAPPING[obj_type].lower()}s moving {movement_dir_txt} in the camera image?"
            )
            answers.append(answer)

        # Generate all possible combinations for negative sampling
        all_object_types = [
            x
            for x in WAYMO_TYPE_MAPPING
            if x is not None and (x not in ["TYPE_UNKNOWN", "TYPE_SIGN"])
        ]
        all_headings = [h.value for h in HeadingType]

        all_possible_combinations = []
        for obj_type in all_object_types:
            for heading in all_headings:
                for speed_cat, _, _ in OBJECT_SPEED_CATS[obj_type]:
                    all_possible_combinations.append((obj_type, heading, speed_cat))

        # Calculate how many negative samples we need
        negative_samples_needed = int(
            len(questions)
            * self.negative_sample_ratio
            / (1 - self.negative_sample_ratio)
        )

        # Use the existing function to generate negative samples
        positive_combinations = list(direction_counts.keys())
        negative_combinations = self._generate_negative_samples(
            positive_combinations, all_possible_combinations, negative_samples_needed
        )

        # If we can't get enough negatives, reduce positives to maintain ratio
        if len(negative_combinations) < negative_samples_needed:
            # Calculate how many positives we should keep
            max_positives = int(
                len(negative_combinations)
                * (1 - self.negative_sample_ratio)
                / self.negative_sample_ratio
            )
            if len(questions) > max_positives:
                # Randomly sample from positive examples
                positive_samples = list(zip(questions, answers))
                selected_positives = random.sample(positive_samples, max_positives)
                questions, answers = zip(*selected_positives)
                questions, answers = list(questions), list(answers)

        # Add the negative samples using the generated combinations
        for obj_type, movement_dir_txt, movement_status in negative_combinations:
            questions.append(
                f"Are there any {movement_status} {WAYMO_TYPE_MAPPING[obj_type].lower()}s moving {movement_dir_txt} in the camera image?"
            )
            answers.append("no")

        return questions, answers

    def _object_facing_type(self, camera, frame):
        """Generates binary questions about object type, movement status (moving or static) and heading (facing) combinations."""
        heading_counts = defaultdict(int)

        for obj in frame.objects:
            if obj.type in ["TYPE_UNKNOWN", "TYPE_SIGN"]:
                continue

            if not obj.is_visible_on_camera(frame, camera):
                continue

            heading_txt = obj.get_camera_heading_direction(frame, camera)
            if heading_txt is None:
                continue

            obj_type = obj.type
            if obj_type is None:
                continue

            heading = obj.get_camera_heading_direction(frame, camera)
            movement_status = obj.get_speed_category()
            key = (obj_type, heading_txt, movement_status)
            heading_counts[key] += 1

        questions = []
        answers = []

        # Add positive examples (where objects exist)
        for (obj_type, heading, movement_status), count in heading_counts.items():
            answer = "yes" if count > 0 else "no"
            questions.append(
                f"Are there any {movement_status} {WAYMO_TYPE_MAPPING[obj_type].lower()}s facing {heading} in the camera image?"
            )
            answers.append(answer)

        # Generate all possible combinations for negative sampling
        all_object_types = [
            x
            for x in WAYMO_TYPE_MAPPING
            if x is not None and (x not in ["TYPE_UNKNOWN", "TYPE_SIGN"])
        ]
        all_headings = [h.value for h in HeadingType]

        all_possible_combinations = []
        for obj_type in all_object_types:
            for heading in all_headings:
                for speed_cat, _, _ in OBJECT_SPEED_CATS[obj_type]:
                    all_possible_combinations.append((obj_type, heading, speed_cat))

        # Calculate how many negative samples we need
        negative_samples_needed = int(
            len(questions)
            * self.negative_sample_ratio
            / (1 - self.negative_sample_ratio)
        )

        # Use the existing function to generate negative samples
        positive_combinations = list(heading_counts.keys())
        negative_combinations = self._generate_negative_samples(
            positive_combinations, all_possible_combinations, negative_samples_needed
        )

        # If we can't get enough negatives, reduce positives to maintain ratio
        if len(negative_combinations) < negative_samples_needed:
            # Calculate how many positives we should keep
            max_positives = int(
                len(negative_combinations)
                * (1 - self.negative_sample_ratio)
                / self.negative_sample_ratio
            )
            if len(questions) > max_positives:
                # Randomly sample from positive examples
                positive_samples = list(zip(questions, answers))
                selected_positives = random.sample(positive_samples, max_positives)
                questions, answers = zip(*selected_positives)
                questions, answers = list(questions), list(answers)

        # Add the negative samples using the generated combinations
        for obj_type, heading, movement_status in negative_combinations:
            questions.append(
                f"Are there any {movement_status} {WAYMO_TYPE_MAPPING[obj_type].lower()}s facing {heading} in the camera image?"
            )
            answers.append("no")

        return questions, answers

    def _generate_negative_samples(self, positive_choices, all_choices, count_needed):
        """Generate negative samples from choices not in positive set."""
        negative_choices = [
            choice for choice in all_choices if choice not in positive_choices
        ]
        num_to_sample = min(count_needed, len(negative_choices))
        return (
            random.sample(negative_choices, num_to_sample) if negative_choices else []
        )

    def generate(
        self, frames
    ) -> List[Tuple[SingleImageMultipleChoiceQuestion, MultipleChoiceAnswer]]:
        """Generate binary (yes/no) questions."""
        samples = []

        frame = random.choice(frames)

        for camera in frame.cameras:
            # Generate questions using generators
            for question_func in self.prompt_functions:
                questions, answers = question_func(camera, frame)

                for question_text, answer_text in zip(questions, answers):
                    question = SingleImageMultipleChoiceQuestion(
                        image_path=camera.image_path,
                        question=question_text,
                        scene_id=frame.scene_id,
                        timestamp=frame.timestamp,
                        camera_name=camera.name,
                        generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                        choices=["yes", "no"],
                    )

                    answer = MultipleChoiceAnswer(
                        choices=["yes", "no"], answer=answer_text
                    )
                    samples.append((question, answer))

        return samples

    def visualise_sample(
        self,
        question_obj: SingleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        figsize=(12, 8),
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Simple visualization showing question, answer, and 2D bounding boxes for relevant objects.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: MultipleChoiceAnswer object with yes/no answer
            save_path: Path to save the visualization
            frames
            figsize: Figure size (width, height)
            text_fontsize: Font size for question/answer text
            title_fontsize: Font size for the title
            dpi: DPI for saved image
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np
        from pathlib import Path
        import textwrap

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Load and display the image
        try:
            img = Image.open(question_obj.image_path)
            img_array = np.array(img)
            ax.imshow(img_array)

        except Exception as e:
            # If image can't be loaded, show error
            ax.text(
                0.5,
                0.5,
                f"Image not found:\n{question_obj.image_path}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.axis("off")

        # Extract answer info
        answer_text = answer_obj.answer.lower()

        # Set colors based on answer
        if answer_text == "yes":
            answer_color = "green"
            answer_symbol = "✓"
        else:
            answer_color = "red"
            answer_symbol = "✗"

        # Add text overlay for question
        question_text = textwrap.fill(question_obj.question, width=60)
        ax.text(
            0.02,
            0.98,
            f"Q: {question_text}",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )

        # Add answer overlay
        answer_display = f"A: {answer_text.upper()} {answer_symbol}"
        ax.text(
            0.02,
            0.02,
            answer_display,
            transform=ax.transAxes,
            fontsize=text_fontsize + 2,
            weight="bold",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=answer_color,
                alpha=0.8,
                edgecolor=answer_color,
            ),
        )

        # Title
        camera_name = getattr(question_obj, "camera_name", "Unknown")
        plt.title(
            f"ObjectBinaryPromptGenerator - {camera_name}",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def get_question_type(self):
        return Union[SingleImageQuestion, MultipleImageQuestion]

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer
