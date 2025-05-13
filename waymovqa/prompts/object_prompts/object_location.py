import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
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
from waymovqa.prompts.base import BasePromptGenerator
from waymovqa.prompts import register_prompt_generator

@register_prompt_generator
class ObjectLocationPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Dict[str, Any]]:
        """Generate location-based questions."""
        raise NotImplementedError()
    
    @abstractmethod
    def get_metric_class(self) -> str:
        return ''
    
    @abstractmethod
    def get_answer_type(self):
        return None
