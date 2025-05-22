import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.questions.single_image import SingleImageQuestion
from waymovqa.questions.multi_image import MultipleImageQuestion
from waymovqa.answers.regression import RegressionAnswer
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator


class SingleImageObjectLocationPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations relative to the ego vehicle."""

    object_ids: Set

    def __init__(self) -> None:
        super().__init__()

        self.QUESTION_CONFIGS = [
            (self._obj_distance, "How far away is the {}?"),
            (
                self._obj_angle,
                "At what horizontal angle is the {} from the ego vehicle?",
            ),
            (
                self._obj_heading_angle,
                "What is the heading angle (in degrees) of the {} relative to the camera image?",
            ),
        ]

    def _obj_distance(
        self, frame: FrameInfo, camera: CameraInfo, obj: ObjectInfo
    ) -> float:
        """Returns the distance to the object."""
        obj_centre_cam = camera.project_to_camera_xyz(obj.get_centre().reshape(1, 3))

        return np.linalg.norm(obj_centre_cam).item()

    def _obj_angle(
        self, frame: FrameInfo, camera: CameraInfo, obj: ObjectInfo
    ) -> float:
        """Returns the angle to the object in camera coordinates."""

        # positive X -> forward, positive y -> left
        obj_pos_cam = camera.project_to_camera_xyz(
            obj.get_centre().reshape(1, 3)
        ).reshape(3)

        angle = np.rad2deg(np.arctan2(obj_pos_cam[0, 1], obj_pos_cam[0]))

        return angle

    def _obj_heading_angle(
        self, frame: FrameInfo, camera: CameraInfo, obj: ObjectInfo
    ) -> float:
        """Returns the angle to the object in camera coordinates."""

        obj_centre = obj.get_centre().reshape(1, 3)

        box = obj.camera_synced_box if obj.camera_synced_box is not None else obj.box
        # Create a point slightly ahead of the object in its heading direction
        heading_angle = box["heading"]

        heading_vector = np.array(
            [np.cos(heading_angle), np.sin(heading_angle), 0.0]
        ).reshape(1, 3)

        # Scale the heading vector to a reasonable length
        heading_vector = heading_vector * box["length"] * 0.5

        # Create a new point by adding the heading vector to the object's position
        ahead_point = obj_centre + heading_vector

        # Project the points to camera coordinates
        obj_centre_cam = camera.project_to_camera_xyz(obj_centre).reshape(3)
        ahead_point_cam = camera.project_to_camera_xyz(ahead_point).reshape(3)

        # Calculate movement vector in camera coordinates
        vector = ahead_point_cam - obj_centre_cam
        vector = vector / np.linalg.norm(vector)

        angle = np.arctan2(vector[1], vector[0])

        return angle

    def _generate_prompts(self, object_best_views):
        """Generate location-based questions."""
        samples = []
        for frame, camera, obj in object_best_views:
            if obj.id in self.object_ids:
                continue

            object_description = obj.get_object_description()

            if object_description is None:
                continue

            other_object_descriptions = [
                obj.get_object_description()
                for obj in frame.objects
                if obj.id != obj.id
            ]

            if object_description in other_object_descriptions:
                continue

            for reg_func, template in self.QUESTION_CONFIGS:
                question = template.format(object_description)

                reg_val = reg_func(frame, camera, obj)

                # Create and append the question-answer pair
                samples.append(
                    (
                        SingleImageQuestion(
                            image_path=camera.image_path,
                            question=question,
                            camera_name=camera.name,
                            scene_id=frame.scene_id,
                            timestamp=frame.timestamp,
                            generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                        ),
                        RegressionAnswer(value=reg_val),
                    )
                )

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
            f"ObjectLocationPromptGenerator - {camera_name}",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def get_metric_class(self) -> str:
        return "RegressionMetric"

    def get_question_type(self) -> type:
        return SingleImageQuestion

    def get_answer_type(self):
        return RegressionAnswer
