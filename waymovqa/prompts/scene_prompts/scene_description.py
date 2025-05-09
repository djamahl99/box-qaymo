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
class SceneDescriptionPromptGenerator(BasePromptGenerator):
    """Generates questions about overall scene description."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Dict[str, Any]]:
        """Generate scene description questions."""
        samples = []

        # Filter for objects with labels
        labeled_objects = [obj for obj in objects if obj.cvat_label]

        if not labeled_objects:
            return []

        # Choose timestamp to focus on if frame not provided
        if frame:
            timestamp = frame.timestamp
        elif labeled_objects and labeled_objects[0].frames:
            timestamp = random.choice(labeled_objects[0].frames)
        else:
            return []

        # Get objects visible in the chosen timestamp
        visible_objects = [obj for obj in labeled_objects if timestamp in obj.frames]

        if not visible_objects:
            return []

        # Group objects by label and count
        object_counts = defaultdict(int)
        for obj in visible_objects:
            if obj.cvat_label:
                object_counts[obj.cvat_label.lower()] += 1

        # Generate scene description question
        if object_counts:
            question = "What objects are present in the scene?"

            # Create answer listing objects and counts
            objects_list = []
            for label, count in object_counts.items():
                if count > 1:
                    objects_list.append(f"{count} {label}s")
                else:
                    objects_list.append(f"a {label}")

            # Format the list with commas and "and"
            if len(objects_list) == 1:
                objects_str = objects_list[0]
            elif len(objects_list) == 2:
                objects_str = f"{objects_list[0]} and {objects_list[1]}"
            else:
                objects_str = ", ".join(objects_list[:-1]) + f", and {objects_list[-1]}"

            answer = f"The scene contains {objects_str}."

            samples.append(
                {
                    "question": question,
                    "answer": answer,
                    "scene_id": scene.scene_id,
                    "timestamp": timestamp,
                    "object_ids": [obj.id for obj in visible_objects],
                    "question_type": "scene_description",
                    "metadata": {
                        "object_counts": {k: v for k, v in object_counts.items()},
                        "total_objects": len(visible_objects),
                    },
                }
            )

        return samples
