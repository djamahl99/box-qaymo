import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.answers.raw_text import RawTextAnswer
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator

@register_prompt_generator
class ObjectRelationPromptGenerator(BasePromptGenerator):
    """Generates questions about relationships between objects."""
    
    def generate(self, scene, objects, frame, frames):
        """Generate relationship questions."""
        raise NotImplementedError()
    
    def get_metric_class(self) -> str:
        return 'MultipleChoiceMetric'
    
    def get_answer_type(self):
        return MultipleChoiceAnswer
    
    def get_supported_methods(self):
        return [] # TODO: nothing yet.