import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.single_image_single_object import (
    SingleImageSingleObjectQuestion,
)
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator


from waymovqa.primitives import colors as labelled_colors


@register_prompt_generator
class ObjectColorPromptGenerator(BasePromptGenerator):
    """Generates questions about object colors."""

    def generate(
        self, frames
    ) -> List[Tuple[SingleImageSingleObjectQuestion, MultipleChoiceAnswer]]:
        """Generate color-based questions."""
        samples = []

        # Track all objects across frames
        object_best_views = self._find_best_object_views(frames)

        # Generate color questions
        for frame, camera, obj in object_best_views.values():
            if obj.cvat_label and obj.cvat_color and camera is not None:
                question = f"What color is the {obj.cvat_label.lower()}?"  # TODO: add more specific prompt?

                question = SingleImageSingleObjectQuestion(
                    image_path=camera.image_path,
                    question=question,
                    object_id=obj.id,
                    scene_id=obj.scene_id,
                    timestamp=frame.timestamp,
                    camera_name=camera.name,
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                )
                answer = MultipleChoiceAnswer(
                    choices=[x.lower() for x in labelled_colors],
                    answer=obj.cvat_color.lower(),
                )

                samples.append((question, answer))

        return samples

    def get_question_type(self) -> type:
        return SingleImageSingleObjectQuestion

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer
