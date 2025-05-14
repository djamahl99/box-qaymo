import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.metrics.coco import COCOMetric
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator
from waymovqa.answers import Object2DAnswer, MultiObject2DAnswer, BaseAnswer
from waymovqa.questions import SingleImageQuestion, MultiPromptSingleImageQuestion
from waymovqa.questions.multi_prompt_single_image import PromptEntry

MIN_OBJECT_SIZE = 32

def draw_3d_wireframe_box_cv(img, u, v, color, thickness=3):
    """Draws 3D wireframe bounding boxes onto the given image."""
    # List of lines to interconnect. Allows for various forms of connectivity.
    # Four lines each describe bottom face, top face, and vertical connectors.
    lines = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))

    for (point_idx1, point_idx2) in lines:

        pt1 = (u[point_idx1], v[point_idx1])
        pt2 = (u[point_idx2], v[point_idx2])

        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        img = cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


    return img

_DEBUG = True

@register_prompt_generator
class Grounding2DPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations with multiple prompts per image."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo
    ) -> List[Tuple[MultiPromptSingleImageQuestion, BaseAnswer]]:
        samples = []

        labeled_objects = [
            obj
            for obj in frame.objects
            if obj.cvat_label
        ]

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
                    uvdok = obj.project_to_image(frame_info=frame, camera_info=camera, return_depth=True)
                    u, v, depth, ok = uvdok.transpose()
                    ok = ok.astype(bool)

                    if sum(ok) < 6 or min(depth) < 0:
                        continue

                    width = max(u) - min(u)
                    height = max(v) - min(v)
                    if width < 32 or height < 32:
                        continue

                    x_min, x_max = int(min(u)), int(max(u))
                    y_min, y_max = int(min(v)), int(max(v))

                    boxes.append([x_min, y_min, x_max, y_max])
                    object_ids.append(obj.id)

                if boxes:
                    prompt_entry = PromptEntry(
                        prompt=prompt,
                        object_ids=object_ids,
                        answers=[Object2DAnswer(box=box, score=1.0, prompt=prompt, object_id=obj_id) for box, obj_id in zip(boxes, object_ids)]
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
                            img_vis, prompt, (x1, y1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                        )
                
                    cv2.imwrite(f'grounding.jpg', img_vis)

            if not prompt_entries:
                continue

            multi_prompt_question = MultiPromptSingleImageQuestion(
                image_path=camera.image_path,
                scene_id=scene.scene_id,
                timestamp=timestamp,
                camera_name=camera.name,
                prompts=prompt_entries,
            )

            samples.append((multi_prompt_question, None))  # or attach answers if needed

        return samples

    # def generate(
    #     self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    # ) -> List[Tuple[SingleImageQuestion, Object2DAnswer]]:
    #     """Generate location-based questions."""
    #     samples = []
    #     sample_id = 0

    #     # Filter for objects with CVAT labels and sufficient lidar points
    #     labeled_objects = [
    #         obj
    #         for obj in objects
    #         if obj.cvat_label and frame.timestamp in obj.frames and obj.camera_synced_box is not None and obj.num_top_lidar_points_in_box > 10
    #     ]

    #     if not labeled_objects:
    #         return []

    #     # Select target object
    #     target_obj = random.choice(labeled_objects)

    #     # Select target timestamp
    #     timestamp = frame.timestamp

    #     if not target_obj.camera_synced_box:
    #         return []

    #     # Get camera by visible
    #     camera_name = target_obj.most_visible_camera_name

    #     # Get frame camera
    #     camera = None
    #     for cam in frame.cameras:
    #         if cam.name == camera_name:
    #             camera = cam
    #             break
        
    #     if camera is None:
    #         return []

    #     if target_obj.cvat_color is not None and target_obj.cvat_label is not None:
    #         question = f"{target_obj.cvat_color} {target_obj.cvat_label.lower()}"
            
    #     elif target_obj.cvat_label:
    #         question = f"{target_obj.cvat_label.lower()}"
    #     else:
    #         return [] # TODO: should we use objects that have not been labeled in cvat?

    #     # Create answer based on object position
    #     uvdok = target_obj.project_to_image(frame_info=frame, camera_info=camera, return_depth=True)
        
    #     u, v, depth, ok = uvdok.transpose()
    #     ok = ok.astype(bool)

    #     # Skip if not all corners visible
    #     if sum(ok) < 6:
    #         return []
        
    #     # The depth should all be in front of the camera
    #     if min(depth) < 0:
    #         return []

    #     # Filter low-quality views
    #     # Skip objects that are too small
    #     width = max(u) - min(u)
    #     height = max(v) - min(v)
    #     if (width < sum(ok) < 32 or height < sum(ok) < 32):
    #         return []
            
    #     x_min, x_max = int(min(u)), int(max(u))
    #     y_min, y_max = int(min(v)), int(max(v))

    #     bbox = [x_min, y_min, x_max, y_max]
    #     # bbox = [y_min, x_min, y_max, x_max]

    #     question = SingleImageQuestion(
    #         image_path=camera.image_path, 
    #         question=question,
    #         object_id=target_obj.id,
    #         scene_id=target_obj.scene_id,
    #         timestamp=timestamp,
    #         camera_name=camera.name
    #     )
    #     answer = Object2DAnswer(box=bbox, score=1.0) # GT has 1.0 score
    #     # TODO: add multiple answers for any objects that fit that prompt...

    #     # DEBUGGING
    #     dataset_path = Path("/media/local-data/uqdetche/waymo_vqa_dataset/")
    #     img_path = dataset_path / question.image_path
    #     img_vis = cv2.imread(img_path)

    #     x1, y1, x2, y2 = bbox
    #     color = (0, 255, 0)
    #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)

    #     draw_3d_wireframe_box_cv(img_vis, u, v, color=(255, 0, 0))
        
    #     # Add confidence text
    #     text = question.question
    #     cv2.putText(
    #         img_vis, text, (x1, y1), 
    #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
    #     )
        
    #     cv2.imwrite(f'grounding_{sample_id}.jpg', img_vis)

    #     samples.append((question, answer))
    #     sample_id +=1

    #     return samples
    
    def get_question_type(self):
        return Object2DAnswer

    def get_answer_type(self) -> type:
        return SingleImageQuestion
    
    def get_metric_class(self) -> str:
        return COCOMetric

print(f"Registered prompt generator: {Grounding2DPromptGenerator.__name__}")