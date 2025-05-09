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
class ObjectLocationPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations."""

    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Dict[str, Any]]:
        """Generate location-based questions."""
        samples = []

        # Filter for objects with CVAT labels and sufficient lidar points
        labeled_objects = [
            obj
            for obj in objects
            if obj.cvat_label and obj.num_lidar_points_in_box > 10
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

        # Get objects visible in the chosen timestamp
        visible_objects = [obj for obj in labeled_objects if timestamp in obj.frames]

        if len(visible_objects) < 2:
            return []

        # Select target object
        target_obj = random.choice(visible_objects)

        # Generate "Where is X?" question
        if target_obj.cvat_label:
            question = f"Where is the {target_obj.cvat_label.lower()}?"

            # Create answer based on object position
            if target_obj.box:
                x, y, z = (
                    target_obj.box["center_x"],
                    target_obj.box["center_y"],
                    target_obj.box["center_z"],
                )

                # Get nearby objects for reference
                other_objects = [
                    obj for obj in visible_objects if obj.id != target_obj.id
                ]
                nearest_objects = sorted(
                    other_objects,
                    key=lambda obj: np.sqrt(
                        (obj.box["center_x"] - x) ** 2
                        + (obj.box["center_y"] - y) ** 2
                        + (obj.box["center_z"] - z) ** 2
                    ),
                )[:3]

                if nearest_objects:
                    nearest_obj = nearest_objects[0]
                    # Determine relative position (left/right/in front of/behind)
                    dx = target_obj.box["center_x"] - nearest_obj.box["center_x"]
                    dy = target_obj.box["center_y"] - nearest_obj.box["center_y"]

                    if abs(dx) > abs(dy):
                        direction = "to the right of" if dx > 0 else "to the left of"
                    else:
                        direction = "in front of" if dy > 0 else "behind"

                    answer = f"The {target_obj.cvat_label.lower()} is {direction} the {nearest_obj.cvat_label.lower()}."
                else:
                    # Fallback answer using coordinates
                    answer = f"The {target_obj.cvat_label.lower()} is located at position ({x:.2f}, {y:.2f}, {z:.2f})."

                samples.append(
                    {
                        "question": question,
                        "answer": answer,
                        "scene_id": scene.scene_id,
                        "timestamp": timestamp,
                        "object_id": target_obj.id,
                        "question_type": "location",
                        "metadata": {
                            "target_object": target_obj.id,
                            "target_label": target_obj.cvat_label,
                            "target_position": [x, y, z],
                        },
                    }
                )

        return samples
