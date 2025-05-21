import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict

from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.data.visualise import draw_3d_wireframe_box_cv
from waymovqa.metrics.coco import COCOMetric
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator
from waymovqa.answers import Object2DAnswer, MultiObject2DAnswer, BaseAnswer
from waymovqa.questions import (
    SingleImageSingleObjectQuestion,
    SingleImageMultiplePromptQuestion,
)
from waymovqa.questions.single_image_multi_prompt import PromptEntry

MIN_OBJECT_SIZE = 32


@register_prompt_generator
class Grounding2DPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations with multiple prompts per image."""

    sampled_object_ids: Set[str]

    def generate(
        self,
        frames,
    ) -> List[Tuple[SingleImageMultiplePromptQuestion, BaseAnswer]]:
        samples = []

        # Track all objects across frames to find good instances
        object_best_views = self._find_best_object_views(frames)

        # Map timestamp to frames list index
        timestamp_to_frame = {frame.timestamp: frame for frame in frames}

        frame_cams = {}
        frame_cam_set = set()
        for frame, camera, obj in object_best_views.values():
            frame_cam_set.add((frame.timestamp, camera.name))

            frame_cams.setdefault(frame.timestamp, {})
            frame_cams[frame.timestamp][camera.name] = camera

        # Generate questions for each object using its best frame/camera
        for frame_timestamp, camera_name in frame_cam_set:
            frame = timestamp_to_frame[frame_timestamp]
            camera = frame_cams[frame_timestamp, camera_name]

            prompt_to_objs = defaultdict(list)

            for obj in frame.objects:
                # Build prompt string
                prompt = obj.get_object_description()

                if prompt is None:
                    continue

                prompt_to_objs[prompt].append(obj)

            prompt_entries = []
            for prompt, objs in prompt_to_objs.items():
                boxes, object_ids = [], []

                for obj in objs:
                    uvdok = obj.project_to_image(
                        frame_info=frame, camera_info=camera, return_depth=True
                    )
                    u, v, depth, ok = uvdok.transpose()
                    ok = ok.astype(bool)

                    x_min, x_max = int(min(u)), int(max(u))
                    y_min, y_max = int(min(v)), int(max(v))

                    x_min, x_max = [
                        max(min(x, camera.width), 0) for x in [x_min, x_max]
                    ]
                    y_min, y_max = [
                        max(min(y, camera.height), 0) for y in [y_min, y_max]
                    ]

                    boxes.append([x_min, y_min, x_max, y_max])
                    object_ids.append(obj.id)

                if boxes:
                    prompt_entry = PromptEntry(
                        prompt=prompt,
                        object_ids=object_ids,
                        answers=[
                            Object2DAnswer(
                                box=box, score=1.0, prompt=prompt, object_id=obj_id
                            )
                            for box, obj_id in zip(boxes, object_ids)
                            if obj_id not in self.sampled_object_ids
                        ],
                    )
                    prompt_entries.append(prompt_entry)

            if not prompt_entries:
                continue

            multi_prompt_question = SingleImageMultiplePromptQuestion(
                image_path=camera.image_path,
                scene_id=frame.scene_id,
                timestamp=frame.timestamp,
                camera_name=camera.name,
                prompts=prompt_entries,
                generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
            )

            samples.append((multi_prompt_question, None))

            for prompt_entry in prompt_entries:
                for obj_id in prompt_entry.object_ids:
                    self.sampled_object_ids.add(obj_id)

        return samples

    def get_question_type(self):
        return SingleImageMultiplePromptQuestion

    def get_answer_type(self):
        return Union[Object2DAnswer, MultiObject2DAnswer]

    def get_metric_class(self):
        return COCOMetric
