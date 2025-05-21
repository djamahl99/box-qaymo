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

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.single_image_multi_choice import (
    SingleBase64ImageMultipleChoiceQuestion,
)
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
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


class HeadingType(str, Enum):
    TOWARDS = "towards"
    AWAY = "away"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@register_prompt_generator
class ObjectDrawnBoxPromptGenerator(BasePromptGenerator):
    """Generates questions about objects highlighted in te image."""

    def __init__(self) -> None:
        super().__init__()

        self.ATTRIBUTE_CONFIGS = [
            (labelled_colors, "cvat_color", COLOR_QUESTIONS_CHOICES, None),
            (finegrained_labels, "cvat_label", LABEL_QUESTIONS_CHOICES, None),
            (HeadingType, None, HEADING_QUESTIONS_CHOICES, self._heading_function),
        ]

    def _heading_function(
        self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo
    ) -> str:
        """Returns heading choice -> returns one of 'towards', 'away', 'left', 'right'"""
        if obj.type == "TYPE_SIGN":
            return None

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

        # Normalize the vector for direction determination
        normalized_vector = vector.reshape(3) / np.linalg.norm(vector)

        # Find the dominant direction
        max_axis = np.argmax(np.abs(normalized_vector))

        if max_axis == 0:  # X-axis dominates (left-right)
            if normalized_vector[max_axis] > 0:
                return "away"
            else:
                return "towards"
        elif max_axis == 1:  # Y-axis dominates (up-down)
            if normalized_vector[max_axis] > 0:
                return "left"
            else:
                return "right"

        else:  # Z-axis dominates (towards-away)
            if normalized_vector[max_axis] > 0:
                return "up"
            else:
                return "down"

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
                    image_bas64=img_base64,
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

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer

    def get_question_type(self):
        return SingleBase64ImageMultipleChoiceQuestion
