import enum
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod
import base64
from enum import Enum
import tensorflow as tf

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.single_image_multi_choice import SingleBase64ImageMultipleChoiceQuestion
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator

from waymovqa.data.visualise import draw_3d_wireframe_box_cv, generate_object_depth_buffer
from waymovqa.primitives import colors as labelled_colors
from waymovqa.primitives import labels as finegrained_labels

class HeadingType(str, Enum):
    TOWARDS = "towards"
    AWAY = "away"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = 'down'

@register_prompt_generator
class ObjectDrawnBoxPromptGenerator(BasePromptGenerator):
    """Generates questions about objects highlighted in te image."""

    COLOR_QUESTIONS = [
        "How would you describe the color of the object out of {}?",
        "What color is the highlighted object? Choose from {}.",
        "Select the best color description for this object: {}."
    ]

    LABEL_QUESTIONS = [
        "Classify the image into one of the following: {}.",
        "What type of object is highlighted? Choose from {}.",
        "Identify this object as one of the following: {}."
    ]
    
    HEADING_QUESTIONS = [
        "How would you describe the heading of the highlighted object? Choose from {}."
    ]

    def __init__(self) -> None:
        super().__init__()
        
        self.ATTRIBUTE_CONFIGS = [
            (labelled_colors, "cvat_color", self.COLOR_QUESTIONS, None),
            (finegrained_labels, "cvat_label", self.LABEL_QUESTIONS, None),
            (HeadingType, None, self.HEADING_QUESTIONS, self._heading_function)
        ]


    def _heading_function(self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo) -> str:
        """Returns heading choice -> returns one of 'towards', 'away', 'left', 'right'"""
        if obj.type == 'TYPE_SIGN':
            return None
        
        # Extract object center and create debug info dictionary
        debug_info = {}
        obj_centre = np.array([obj.box['center_x'], obj.box['center_y'], obj.box['center_z']]).reshape(1, 3)
        
        # Create a point slightly ahead of the object in its heading direction
        heading_angle = obj.box['heading']
        
        heading_vector = np.array([
            np.cos(heading_angle),
            np.sin(heading_angle),
            0.0
        ]).reshape(1, 3)
        
        # Scale the heading vector to a reasonable length
        heading_vector = heading_vector * obj.box['length'] * 0.5
        
        # Create a new point by adding the heading vector to the object's position
        ahead_point = obj_centre + heading_vector
        
        # Project the points to camera coordinates
        obj_centre_cam = camera.project_to_camera_xyz(obj_centre).reshape(3)
        ahead_point_cam = camera.project_to_camera_xyz(ahead_point).reshape(3)
        
        # Calculate movement vector in camera coordinates
        vector = ahead_point_cam - obj_centre_cam

        # Normalize the vector for direction determination
        normalized_vector = vector.reshape(3) / np.linalg.norm(vector)
        
        # Find the dominant direction
        max_axis = np.argmax(np.abs(normalized_vector))
        
        if max_axis == 0:  # X-axis dominates (left-right)
            if normalized_vector[max_axis] > 0:
                return 'away'
            else:
                return 'towards'
        elif max_axis == 1:  # Y-axis dominates (up-down)
            if normalized_vector[max_axis] > 0:
                return 'left'
            else:
                return 'right'

        else:  # Z-axis dominates (towards-away)
            if normalized_vector[max_axis] > 0:
                return 'up'
            else:
                return 'down'


    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Tuple[SingleBase64ImageMultipleChoiceQuestion, MultipleChoiceAnswer]]:
        """Generates questions about objects with bounding boxes drawn on the frame."""
        samples = []
        
        assert frame is not None

        for camera in frame.cameras:
            camera_objects = [obj for obj in objects if obj.most_visible_camera_name == camera.name]

            # depth_buffer = generate_object_depth_buffer(frame, camera) # TODO: bring this back?
            img_vis = cv2.imread(camera.image_path) # TODO: remove
            cv2.imwrite('cam.jpg', img_vis)


            for obj in camera_objects:
                # Get camera by visible
                camera_name = obj.most_visible_camera_name

                # Get scene camera
                camera = None
                for cam in frame.cameras:
                    if cam.name == camera_name:
                        camera = cam
                        break

                assert Path(camera.image_path).exists()

                with tf.device("/CPU:0"):
                    uvdok = obj.project_to_image(
                        frame_info=frame, camera_info=camera, return_depth=True
                    )
                    u, v, depth, ok = uvdok.transpose()
                    ok = ok.astype(bool)

                if sum(ok) < 6 or min(depth) < 0:
                    continue

                depth_min = depth[ok].min()

                x_min, x_max = int(min(u)), int(max(u))
                y_min, y_max = int(min(v)), int(max(v))

                x_min, x_max = [max(min(x, camera.width), 0) for x in [x_min, x_max]]
                y_min, y_max = [max(min(y, camera.height), 0) for y in [y_min, y_max]]

                width = x_max - x_min
                height = y_max - y_min
                if width < 32 or height < 32:
                    continue

                # depth_buff_min = depth_buffer[y_min:y_max, x_min:x_max].min() # TODO: check if this is effective?

                # # occluded
                # if depth_buff_min < depth_min:
                #     # TODO: mark as occluded?
                #     continue

                if obj.num_lidar_points_in_box < 5:
                    print('< 5 points')
                    continue

                color = (0, 255, 0)
                img_vis = cv2.imread(camera.image_path)
                cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), color, 4)
                # cv2.putText(
                #     img_vis,
                #     f'{obj.cvat_label} {obj.cvat_color}',
                #     (x_min, y_min),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (255, 255, 255),
                #     2,
                #     cv2.LINE_AA,
                # )

                # cv2.imwrite('object_drawn_box_prompt.jpg', img_vis)

                _, buffer = cv2.imencode('.jpg', img_vis)  # or '.png'
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                for choices, attr_name, question_templates, question_func in self.ATTRIBUTE_CONFIGS:
                    if isinstance(choices, type) and issubclass(choices, enum.Enum):
                        choices = [enum_val.value for enum_val in choices] # replace with list of choices   
                        
                    if question_func is not None:
                       answer_txt = question_func(obj=obj, camera=camera, frame=frame)
                    else:
                        answer_txt = getattr(obj, attr_name)
                    
                    if answer_txt is None or answer_txt not in choices:
                        continue
                        
                    # Format options for all question texts
                    formatted_options = ', '.join(choices[:-1]) + ' or ' + choices[-1]
                    
                    # Select one question template randomly
                    question_template = random.choice(question_templates)
                    
                    # Create the question with formatted options
                    question_text = question_template.format(formatted_options)
                    
                    cv2.putText(
                        img_vis,
                        answer_txt,
                        (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imwrite('object_drawn_box_prompt.jpg', img_vis)
                    
                    question = SingleBase64ImageMultipleChoiceQuestion(
                        image_bas64=img_base64,
                        question=question_text,
                        choices=choices,
                        scene_id=obj.scene_id,
                        timestamp=frame.timestamp,
                        camera_name=camera.name
                    )
                    
                    answer = MultipleChoiceAnswer(
                        choices=choices,
                        answer=answer_txt,
                    )
                    
                    samples.append((question, answer))

        return samples

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer
    
    def get_supported_methods(self) -> List[str]:
        return ['frame', 'object']
    
    def get_question_type(self):
        return SingleBase64ImageMultipleChoiceQuestion