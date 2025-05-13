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
from waymovqa.prompts.base import BasePromptGenerator
from waymovqa.prompts import register_prompt_generator
from waymovqa.answers import Object2DAnswer
from waymovqa.questions import SingleImageQuestion

MIN_OBJECT_SIZE = 32

@register_prompt_generator
class Grounding2DPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Tuple[SingleImageQuestion, Object2DAnswer]]:
        """Generate location-based questions."""
        samples = []

        # Filter for objects with CVAT labels and sufficient lidar points
        labeled_objects = [
            obj
            for obj in objects
            if obj.cvat_label #and obj.num_lidar_points_in_box > 10
        ]

        if not labeled_objects:
            return []

        # Choose timestamp to focus on if frame not provided
        if frame:
            timestamp = frame.timestamp
        elif labeled_objects and labeled_objects[0].frames:
            timestamp = random.choice(labeled_objects[0].frames)
        else:
            return []

        # Select target object
        target_obj = random.choice(labeled_objects)

        if not target_obj.box:
            return []

        # Get camera by visible
        camera_name = target_obj.visible_cameras[0]

        # Get frame camera
        camera = None
        for cam in frame.cameras:
            if cam.name == camera_name:
                camera = cam
                break

        if camera is None:
            return []

        if target_obj.cvat_color is not None and target_obj.cvat_label is not None:
            question = f"{target_obj.cvat_color} {target_obj.cvat_label.lower()}"
            
        elif target_obj.cvat_label:
            question = f"{target_obj.cvat_label.lower()}"
        else:
            return [] # TODO: should we use objects that have not been labeled in cvat?

        # Create answer based on object position
        uvdok = target_obj.project_to_image(frame_info=frame, camera_info=camera, scene_info=scene, return_depth=True)
        
        u, v, depth, ok = uvdok.transpose()
        ok = ok.astype(bool)

        # Skip if not all corners visible
        if sum(ok) < 6:
            return []
        
        # The depth should all be in front of the camera
        if min(depth) < 0:
            return []

        # Filter low-quality views
        # Skip objects that are too small
        width = max(u) - min(u)
        height = max(v) - min(v)
        if (width < sum(ok) < 6 or height < sum(ok) < 6):
            return []
            
        x_min, x_max = int(min(u)), int(max(u))
        y_min, y_max = int(min(v)), int(max(v))

        bbox = [x_min, y_min, x_max, y_max]

        print('camera.image_path', camera.image_path)
        question = SingleImageQuestion(
            image_path=camera.image_path, 
            question=question,
            object_id=target_obj.id,
            scene_id=target_obj.scene_id,
            timestamp=timestamp,
            camera_name=camera.name
        )
        answer = Object2DAnswer(box=bbox, score=1.0) # GT has 1.0 score

        samples.append((question, answer))

        return samples
    
    def get_question_type(self):
        return Object2DAnswer

    def get_answer_type(self) -> type:
        return SingleImageQuestion
    
    def get_metric_class(self) -> str:
        return COCOMetric

print(f"Registered prompt generator: {Grounding2DPromptGenerator.__name__}")