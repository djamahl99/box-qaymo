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
from waymovqa.questions import SingleImageSingleObjectQuestion, SingleImageMultiplePromptQuestion
from waymovqa.questions.single_image_multi_prompt import PromptEntry

MIN_OBJECT_SIZE = 32

_DEBUG = True


@register_prompt_generator
class Grounding2DPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations with multiple prompts per image."""

    def generate(
        self, 
        scene: Optional[SceneInfo] = None, 
        objects: Optional[List[ObjectInfo]] = None, 
        frame: Optional[FrameInfo] = None,
        frames: Optional[List[FrameInfo]] = None
    ) -> List[Tuple[SingleImageMultiplePromptQuestion, BaseAnswer]]:
        samples = []

        labeled_objects = [obj for obj in frame.objects if obj.cvat_label]

        if not labeled_objects:
            return []

        timestamp = frame.timestamp

        for camera in frame.cameras:
            dataset_path = Path("/media/local-data/uqdetche/waymo_vqa_dataset/")
            img_path = dataset_path / camera.image_path
            img_vis = cv2.imread(img_path)

            prompt_to_objs = defaultdict(list)

            for obj in labeled_objects:
                if camera.name not in obj.visible_cameras:
                    continue

                # Build prompt string
                if obj.cvat_color and obj.cvat_label:
                    prompt = f"{obj.cvat_color} {obj.cvat_label.lower()}"
                else:
                    prompt = obj.cvat_label.lower()

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

                    if sum(ok) < 6 or min(depth) < 0:
                        continue

                    x_min, x_max = int(min(u)), int(max(u))
                    y_min, y_max = int(min(v)), int(max(v))

                    x_min, x_max = [max(min(x, camera.width), 0) for x in [x_min, x_max]]
                    y_min, y_max = [max(min(y, camera.height), 0) for y in [y_min, y_max]]

                    width = x_max - x_min
                    height = y_max - y_min
                    if width < 32 or height < 32:
                        continue

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
                        ],
                    )
                    prompt_entries.append(prompt_entry)

                    # TODO: remove
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        color = (0, 255, 0)
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)

                        draw_3d_wireframe_box_cv(img_vis, u, v, color=(255, 0, 0))

                        # Add confidence text
                        cv2.putText(
                            img_vis,
                            prompt,
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.imwrite(f"grounding.jpg", img_vis)

            if not prompt_entries:
                continue

            multi_prompt_question = SingleImageMultiplePromptQuestion(
                image_path=camera.image_path,
                scene_id=scene.scene_id,
                timestamp=timestamp,
                camera_name=camera.name,
                prompts=prompt_entries,
                generator_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            )

            samples.append((multi_prompt_question, None))  # or attach answers if needed

        return samples

    def get_question_type(self):
        return SingleImageMultiplePromptQuestion

    def get_answer_type(self):
        return Union[Object2DAnswer, MultiObject2DAnswer]

    def get_metric_class(self):
        return COCOMetric
    
    def get_supported_methods(self) -> List[str]:
        return ['frame', 'object']
