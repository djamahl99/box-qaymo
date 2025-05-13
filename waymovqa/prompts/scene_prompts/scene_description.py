import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.data import *
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.prompts import BasePromptGenerator, register_prompt_generator

from waymovqa.questions.multi_image import MultipleImageQuestion
from waymovqa.answers.raw_text import RawTextAnswer


@register_prompt_generator
class SceneDescriptionPromptGenerator(BasePromptGenerator):
    """Generates questions about overall scene description."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Dict[str, Any]]:
        """Generate scene description questions."""
        raise NotImplementedError

    @abstractmethod
    def get_question_type(self):
        return MultipleImageQuestion

    @abstractmethod
    def get_anser_type(self):
        return RawTextAnswer
