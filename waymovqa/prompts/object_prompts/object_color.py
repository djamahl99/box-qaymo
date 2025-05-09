import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from waymovqa.data import *
from ..base import BasePromptGenerator, register_prompt_generator


@register_prompt_generator
class ObjectColorPromptGenerator(BasePromptGenerator):
    """Generates questions about object colors."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Dict[str, Any]]:
        """Generate color-based questions."""
        samples = []

        # Filter for objects with colors
        colored_objects = [obj for obj in objects if obj.cvat_color]

        if not colored_objects:
            return []

        # Choose timestamp to focus on if frame not provided
        if frame:
            timestamp = frame.timestamp
        elif colored_objects and colored_objects[0].frames:
            timestamp = random.choice(colored_objects[0].frames)
        else:
            return []

        # Get colored objects visible in the chosen timestamp
        visible_objects = [obj for obj in colored_objects if timestamp in obj.frames]

        if not visible_objects:
            return []

        # Generate color questions
        for obj in random.sample(visible_objects, min(3, len(visible_objects))):
            if obj.cvat_label and obj.cvat_color:
                question = f"What color is the {obj.cvat_label.lower()}?"
                answer = f"The {obj.cvat_label.lower()} is {obj.cvat_color.lower()}."

                samples.append(
                    {
                        "question": question,
                        "answer": answer,
                        "scene_id": scene.scene_id,
                        "timestamp": timestamp,
                        "object_id": obj.id,
                        "question_type": "color",
                        "metadata": {
                            "target_object": obj.id,
                            "target_label": obj.cvat_label,
                            "color": obj.cvat_color,
                        },
                    }
                )

        return samples
