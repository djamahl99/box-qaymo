import enum
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.answers.regression import RegressionAnswer
from waymovqa.questions.single_image import SingleImageQuestion
from waymovqa.questions.multi_image import MultipleImageQuestion
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import WAYMO_TYPE_MAPPING, HeadingType, ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator
from waymovqa.prompt_generators.templates import *
from waymovqa.primitives import WAYMO_LABEL_TYPES

from functools import partial


@register_prompt_generator
class ObjectCountPromptGenerator(BasePromptGenerator):
    """Generates questions about object counts and headings."""

    def __init__(self) -> None:
        super().__init__()

        self.SINGLE_CAMERA_ATTRIBUTE_CONFIGS = [
            (
                HeadingType,
                self._count_heading_camera,
                HEADING_QUESTIONS_COUNT_SINGLE_IMAGE,
            ),
            (
                [x for x in WAYMO_TYPE_MAPPING if x is not None],
                self._count_waymo_type,
                WAYMO_TYPE_COUNT_QUESTIONS,
            ),
        ]

        self.other_generators = [
            self._count_waymo_type_heading,
        ]

    def _count_waymo_type_heading(self, camera, frame):
        """Counts the number of objects per type and heading combination."""
        heading_counts = defaultdict(int)

        for obj in frame.objects:
            if not obj.is_visible_on_camera(frame, camera):
                continue

            if not obj.get_camera_heading_text(camera, frame):
                continue

            obj_type = obj.get_simple_type()
            if obj_type == "Sign":
                continue  # signs don't really have headings

            heading = obj.get_camera_heading_text(camera, frame)

            # Use tuple instead of string concatenation to avoid comma issues
            key = (obj_type, heading)
            heading_counts[key] += 1

        questions = []
        answers = []

        # Fixed: use .items() to get both key and value
        for (obj_type, heading), count in heading_counts.items():
            if count > 0:  # Only generate questions for non-zero counts
                questions.append(
                    f"How many {obj_type}s are heading {heading} in the camera image?"
                )
                answers.append(count)

        return questions, answers

    def _count_heading_camera(self, camera, frame):
        """Count the number of objects per heading."""
        heading_counts = defaultdict(int)

        for obj in frame.objects:
            if not obj.get_camera_heading_text(camera, frame):
                continue

            if obj.get_simple_type() == "Sign":
                continue  # signs don't really have headings

            heading = obj.get_camera_heading_text(camera, frame)
            heading_counts[heading] += 1

        return heading_counts

    def _count_waymo_type(self, camera, frame):
        """Count the number of waymo object types."""
        type_counts = defaultdict(int)

        for obj in frame.objects:
            if not obj.get_camera_heading_text(camera, frame):
                continue

            object_type = obj.get_simple_type()
            if object_type is not None:
                type_counts[object_type] += 1

        return type_counts

    def generate(
        self, frames
    ) -> List[
        Tuple[Union[MultipleImageQuestion, SingleImageQuestion], RegressionAnswer]
    ]:
        """Generate count-based questions."""
        samples = []

        frame = random.choice(frames)

        for camera in frame.cameras:
            # Generate questions using configured templates
            for (
                choices,
                question_func,
                question_templates,
            ) in self.SINGLE_CAMERA_ATTRIBUTE_CONFIGS:
                if isinstance(choices, type) and issubclass(choices, enum.Enum):
                    choices = [enum_val.value for enum_val in choices]

                if question_func is None:
                    continue

                choice_value_dict = question_func(camera, frame)

                for choice, value in choice_value_dict.items():
                    # Validate choice is in expected set
                    if choice not in choices:
                        continue  # Skip invalid choices instead of asserting

                    if value == 0:
                        continue  # Skip zero counts to reduce noise

                    # Select one question template randomly
                    question_template = random.choice(question_templates)

                    # Create the question with the prompted attribute choice
                    question_text = question_template.format(choice)

                    question = SingleImageQuestion(
                        image_path=camera.image_path,
                        question=question_text,
                        scene_id=frame.scene_id,
                        timestamp=frame.timestamp,
                        camera_name=camera.name,
                        generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                    )

                    answer = RegressionAnswer(value=value)
                    samples.append((question, answer))

            # Generate questions using other generators
            for question_func in self.other_generators:
                try:
                    questions, answers = question_func(camera, frame)

                    for question_text, answer_value in zip(questions, answers):
                        question = SingleImageQuestion(
                            image_path=camera.image_path,
                            question=question_text,
                            scene_id=frame.scene_id,
                            timestamp=frame.timestamp,
                            camera_name=camera.name,
                            generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                        )

                        answer = RegressionAnswer(value=answer_value)
                        samples.append((question, answer))
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error in question generator {question_func.__name__}: {e}")
                    continue

        return samples

    def visualise_sample(
        self,
        question_obj: SingleImageQuestion,
        answer_obj: RegressionAnswer,
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
            answer_obj: RegressionAnswer
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
        answer_value = answer_obj.value

        # Format answer based on magnitude
        if answer_value == int(answer_value):
            answer_text = f"{int(answer_value)}"
        else:
            answer_text = f"{answer_value:.2f}"
        answer_color = "green"

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
        answer_display = f"A: {answer_text}"
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
            f"ObjectCountPromptGenerator - {camera_name}",
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
        return "RegressionMetric"  # Changed from MultipleChoiceMetric since answers are counts

    def get_answer_type(self):
        return RegressionAnswer
