import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.single_image import SingleImageQuestion
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompts.base import BasePromptGenerator
from waymovqa.prompts import register_prompt_generator


from waymovqa.primitives import colors as labelled_colors


@register_prompt_generator
class ObjectColorPromptGenerator(BasePromptGenerator):
    """Generates questions about object colors."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Tuple[SingleImageQuestion, MultipleChoiceAnswer]]:
        """Generate color-based questions."""
        samples = []

        # Filter for objects with colors
        colored_objects = [obj for obj in objects if obj.cvat_color]

        if frame is not None:
            # Filter objects that occur in the given timestamp
            colored_objects = [obj for obj in objects if frame.timestamp in obj.frames]

        if not colored_objects:
            return []

        # Get colored objects visible in the chosen timestamp
        visible_objects = [obj for obj in colored_objects]

        if not visible_objects:
            return []

        # Generate color questions
        for obj in random.sample(visible_objects, min(3, len(visible_objects))):
            # Get camera by visible
            camera_name = obj.visible_cameras[0]

            # Get scene camera
            camera = None
            for cam in scene.camera_calibrations:
                if cam.name == camera_name:
                    camera = cam
                    break

            # Choose timestamp to focus on if frame not provided
            if frame:
                timestamp = frame.timestamp
            elif colored_objects and colored_objects[0].frames:
                timestamp = random.choice(colored_objects[0].frames)
            else:
                return []

            if obj.cvat_label and obj.cvat_color and camera is not None:
                question = f"What color is the {obj.cvat_label.lower()}?"  # TODO: add more specific prompt?

                question = SingleImageQuestion(
                    image_path=camera.image_path,
                    question=question,
                    object_id=obj.id,
                    scene_id=obj.scene_id,
                    timestamp=timestamp,
                    camera_name=camera.name
                )
                answer = MultipleChoiceAnswer(
                    choices=[x.lower() for x in labelled_colors],
                    answer=obj.cvat_color.lower(),
                )

                samples.append((question, answer))

        return samples

    @abstractmethod
    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    @abstractmethod
    def get_answer_type(self):
        return MultipleChoiceAnswer