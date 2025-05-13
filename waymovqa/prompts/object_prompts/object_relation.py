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
class ObjectRelationPromptGenerator(BasePromptGenerator):
    """Generates questions about relationships between objects."""
    
    def generate(self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None) -> List[Dict[str, Any]]:
        """Generate relationship questions."""
        samples = []
        
        # Filter for objects with labels
        labeled_objects = [obj for obj in objects if obj.cvat_label]
        
        if len(labeled_objects) < 2:
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
            
        # Select two distinct objects
        sampled_objects = random.sample(visible_objects, min(2, len(visible_objects)))
        if len(sampled_objects) < 2:
            return []
            
        obj1, obj2 = sampled_objects[0], sampled_objects[1]
        
        # Calculate distance between objects
        if obj1.box and obj2.box:
            dx = obj1.box["center_x"] - obj2.box["center_x"]
            dy = obj1.box["center_y"] - obj2.box["center_y"]
            dz = obj1.box["center_z"] - obj2.box["center_z"]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Determine relative position
            if abs(dx) > max(abs(dy), abs(dz)):
                direction = "to the right of" if dx > 0 else "to the left of"
            elif abs(dy) > abs(dz):
                direction = "in front of" if dy > 0 else "behind"
            else:
                direction = "above" if dz > 0 else "below"
            
            # Generate question about relative position
            question = f"What is the relationship between the {obj1.cvat_label.lower()} and the {obj2.cvat_label.lower()}?"
            answer = f"The {obj1.cvat_label.lower()} is {direction} the {obj2.cvat_label.lower()}, approximately {distance:.2f} meters apart."
            
            samples.append({
                "question": question,
                "answer": answer,
                "scene_id": scene.scene_id,
                "timestamp": timestamp,
                "object_ids": [obj1.id, obj2.id],
                "question_type": "relationship",
                "metadata": {
                    "object1": {
                        "id": obj1.id,
                        "label": obj1.cvat_label,
                        "position": [obj1.box["center_x"], obj1.box["center_y"], obj1.box["center_z"]]
                    },
                    "object2": {
                        "id": obj2.id,
                        "label": obj2.cvat_label,
                        "position": [obj2.box["center_x"], obj2.box["center_y"], obj2.box["center_z"]]
                    },
                    "relation": direction,
                    "distance": float(distance)
                }
            })
            
            # Also generate a "which is closer/further" question if there are more objects
            if len(visible_objects) > 2:
                # Find a third object
                remaining_objects = [obj for obj in visible_objects if obj.id not in [obj1.id, obj2.id]]
                if remaining_objects:
                    obj3 = random.choice(remaining_objects)
                    
                    # Calculate distances from obj3 to obj1 and obj2
                    d1 = np.sqrt(
                        (obj3.box["center_x"] - obj1.box["center_x"])**2 +
                        (obj3.box["center_y"] - obj1.box["center_y"])**2 +
                        (obj3.box["center_z"] - obj1.box["center_z"])**2
                    )
                    
                    d2 = np.sqrt(
                        (obj3.box["center_x"] - obj2.box["center_x"])**2 +
                        (obj3.box["center_y"] - obj2.box["center_y"])**2 +
                        (obj3.box["center_z"] - obj2.box["center_z"])**2
                    )
                    
                    # Determine which is closer
                    closer_obj = obj1 if d1 < d2 else obj2
                    farther_obj = obj2 if d1 < d2 else obj1
                    
                    question = f"Is the {obj1.cvat_label.lower()} or the {obj2.cvat_label.lower()} closer to the {obj3.cvat_label.lower()}?"
                    answer = f"The {closer_obj.cvat_label.lower()} is closer to the {obj3.cvat_label.lower()} than the {farther_obj.cvat_label.lower()}."
                    
                    samples.append({
                        "question": question,
                        "answer": answer,
                        "scene_id": scene.scene_id,
                        "timestamp": timestamp,
                        "object_ids": [obj1.id, obj2.id, obj3.id],
                        "question_type": "comparison",
                        "metadata": {
                            "reference_object": {
                                "id": obj3.id,
                                "label": obj3.cvat_label
                            },
                            "closer_object": {
                                "id": closer_obj.id,
                                "label": closer_obj.cvat_label,
                                "distance": float(min(d1, d2))
                            },
                            "farther_object": {
                                "id": farther_obj.id,
                                "label": farther_obj.cvat_label,
                                "distance": float(max(d1, d2))
                            }
                        }
                    })
                
        return samples