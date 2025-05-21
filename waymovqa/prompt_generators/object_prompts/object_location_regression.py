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


@register_prompt_generator
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

    def get_metric_class(self) -> str:
        return "RegressionMetric"

    def get_question_type(self) -> type:
        return SingleImageQuestion

    def get_answer_type(self):
        return RegressionAnswer
