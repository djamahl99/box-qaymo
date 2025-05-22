import enum
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod
import base64
from enum import Enum
from io import BytesIO

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.single_image_multi_choice import (
    SingleBase64ImageMultipleChoiceQuestion,
)
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import HeadingType, ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator

from waymovqa.data.visualise import (
    draw_3d_wireframe_box_cv,
    generate_object_depth_buffer,
)
from waymovqa.primitives import colors as labelled_colors
from waymovqa.primitives import labels as finegrained_labels

from waymovqa.prompt_generators.templates import (
    COLOR_QUESTIONS_CHOICES,
    LABEL_QUESTIONS_CHOICES,
    HEADING_QUESTIONS_CHOICES,
)


@register_prompt_generator
class ObjectDrawnBoxPromptGenerator(BasePromptGenerator):
    """Generates questions about objects highlighted in te image."""

    def __init__(self) -> None:
        super().__init__()

        self.ATTRIBUTE_CONFIGS = [
            (labelled_colors, "cvat_color", COLOR_QUESTIONS_CHOICES, None),
            (finegrained_labels, "cvat_label", LABEL_QUESTIONS_CHOICES, None),
            (
                HeadingType,
                None,
                HEADING_QUESTIONS_CHOICES,
                lambda obj, camera, frame: obj.get_camera_heading_text(camera, frame),
            ),
            # TODO: movement
        ]

    def generate(
        self, frames
    ) -> List[Tuple[SingleBase64ImageMultipleChoiceQuestion, MultipleChoiceAnswer]]:
        """Generates questions about objects with bounding boxes drawn on the frame."""
        samples = []

        # Track all objects across frames
        object_best_views = self._find_best_object_views(frames)

        for frame, camera, obj in object_best_views.values():
            img_vis = cv2.imread(camera.image_path)

            uvdok = obj.project_to_image(
                frame_info=frame, camera_info=camera, return_depth=True
            )
            u, v, depth, ok = uvdok.transpose()
            ok = ok.astype(bool)

            x_min, x_max = int(min(u)), int(max(u))
            y_min, y_max = int(min(v)), int(max(v))

            x_min, x_max = [max(min(x, camera.width), 0) for x in [x_min, x_max]]
            y_min, y_max = [max(min(y, camera.height), 0) for y in [y_min, y_max]]

            color = (0, 255, 0)
            img_vis = cv2.imread(camera.image_path)
            cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), color, 4)

            _, buffer = cv2.imencode(".jpg", img_vis)  # or '.png'
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            for (
                choices,
                attr_name,
                question_templates,
                question_func,
            ) in self.ATTRIBUTE_CONFIGS:
                if isinstance(choices, type) and issubclass(choices, enum.Enum):
                    choices = [enum_val.value for enum_val in choices]

                if question_func is not None:
                    answer_txt = question_func(obj=obj, camera=camera, frame=frame)
                else:
                    answer_txt = getattr(obj, attr_name)

                if answer_txt is None or answer_txt not in choices:
                    continue

                # Format options for all question texts
                formatted_options = ", ".join(choices[:-1]) + " or " + choices[-1]

                # Select one question template randomly
                question_template = random.choice(question_templates)

                # Create the question with formatted options
                question_text = question_template.format(formatted_options)

                question = SingleBase64ImageMultipleChoiceQuestion(
                    image_base64=img_base64,
                    question=question_text,
                    choices=choices,
                    scene_id=obj.scene_id,
                    timestamp=frame.timestamp,
                    camera_name=camera.name,
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                )

                answer = MultipleChoiceAnswer(
                    choices=choices,
                    answer=answer_txt,
                )

                samples.append((question, answer))

        return samples

    def visualise_sample(
        self,
        question_obj: SingleBase64ImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        figsize=(12, 8),
        box_color="green",
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Simple visualization showing question, choices over image with drawn object.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: MultipleChoiceAnswer
            save_path: Path to save the visualization
            frames:
            figsize: Figure size (width, height)
            box_color: Color for bounding boxes
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
        import base64
        from io import BytesIO

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Load and display the image
        try:
            img = Image.open(BytesIO(base64.b64decode(question_obj.image_base64)))
            img_array = np.array(img)
            ax.imshow(img_array)

        except Exception as e:
            # If image can't be loaded, show error
            ax.text(
                0.5,
                0.5,
                f"Image could not be decoded: {str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.axis("off")

        # Get question text
        question_text = question_obj.question

        # Add text overlay for question
        question_text = textwrap.fill(question_text, width=60)
        ax.text(
            0.02,
            0.98,
            f"Q: {question_text}",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )

        # Display choices and highlight the correct answer
        for idx, choice in enumerate(answer_obj.choices):
            if choice == answer_obj.answer:
                choice_color = "lightgreen"
                choice_text = f"✓ {choice} (CORRECT)"
                text_color = "darkgreen"
            else:
                choice_color = "lightcoral"
                choice_text = f"✗ {choice}"
                text_color = "darkred"

            ax.text(
                0.02,
                0.02 + idx * 0.06,  # Better spacing
                choice_text,
                transform=ax.transAxes,
                fontsize=text_fontsize,
                weight="bold",
                verticalalignment="bottom",
                color=text_color,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=choice_color,
                    alpha=0.8,
                    edgecolor=text_color,
                ),
            )

        # Title
        camera_name = getattr(question_obj, "camera_name", "Unknown")
        plt.title(
            f"Object Drawn Box QA - {camera_name}",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer

    def get_question_type(self):
        return SingleBase64ImageMultipleChoiceQuestion
