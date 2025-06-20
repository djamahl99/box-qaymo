import enum
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from box_qaymo.answers.multiple_choice import MultipleChoiceAnswer
from box_qaymo.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from box_qaymo.questions.multi_image import MultipleImageQuestion
from box_qaymo.data.scene_info import SceneInfo
from box_qaymo.data.object_info import (
    OBJECT_SPEED_CATS,
    WAYMO_TYPE_MAPPING,
    DifficultyLevelType,
    HeadingType,
    ObjectInfo,
)
from box_qaymo.data.frame_info import FrameInfo
from box_qaymo.data.camera_info import CameraInfo
from box_qaymo.data.laser_info import LaserInfo
from box_qaymo.prompt_generators.base import DISTANCE_BOUND, MIN_LIDAR_PTS, BasePromptGenerator
from box_qaymo.prompt_generators import register_prompt_generator
from box_qaymo.prompt_generators.templates import *
from box_qaymo.primitives import WAYMO_LABEL_TYPES

from functools import partial

OBJECT_SPEED_CATS_QUESTIONS = {
    "TYPE_UNKNOWN": [],
    "TYPE_VEHICLE": [
        ("stationary", "Are there any stationary vehicles facing {}?"),  # up to 10km/h
        (
            "slow speed",
            "Are there any vehicles travelling at a slow speed facing {}?",
        ),  # up to 40 km/h
        (
            "medium speed",
            "Are there any vehicles travelling at a medium speed facing {}?",
        ),  # up to 70 km/h
        (
            "highway speed",
            "Are there any vehicles travelling at highway speed facing {}?",
        ),
    ],
    "TYPE_PEDESTRIAN": [
        ("stationary", "Are there any stationary pedestrians facing {}?"),
        ("walking", "Are there any pedestrians walking and facing {}?"),
        ("jogging", "Are there any pedestrians jogging and facing {}?"),
        ("running", "Are there any pedestrians running and facing {}?"),
    ],  # jogging about 7mins/km and running faster than 3.0 m/s
    "TYPE_SIGN": [("stationary", "Are there any signs facing {}?")],
    "TYPE_CYCLIST": [
        (
            "stationary",
            "Are there any stationary cyclists facing {}?",
        ),  # up to a bit faster than walking speed
        (
            "slow moving",
            "Are there any slow moving cyclists facing {}?",
        ),  # up to 30 km/h
        (
            "fast",
            "Are there any fast moving cyclists facing {}?",
        ),  # any faster than 30km/h
    ],
}

OBJECT_SPEED_CATS_QUESTIONS_MOVING = {
    "TYPE_UNKNOWN": [],
    "TYPE_VEHICLE": [
        ("stationary", "Are there any stationary vehicles?"),  # up to 10km/h
        (
            "slow speed",
            "Are there any vehicles travelling at a slow speed {}?",
        ),  # up to 40 km/h
        (
            "medium speed",
            "Are there any vehicles travelling at a medium speed {}?",
        ),  # up to 70 km/h
        (
            "highway speed",
            "Are there any vehicles travelling at highway speed {}?",
        ),
    ],
    "TYPE_PEDESTRIAN": [
        ("stationary", "Are there any stationary pedestrians?"),
        ("walking", "Are there any pedestrians walking {}?"),
        ("jogging", "Are there any pedestrians jogging {}?"),
        ("running", "Are there any pedestrians running {}?"),
    ],  # jogging about 7mins/km and running faster than 3.0 m/s
    "TYPE_SIGN": [("stationary", "Are there any signs?")],
    "TYPE_CYCLIST": [
        (
            "stationary",
            "Are there any stationary cyclists?",
        ),  # up to a bit faster than walking speed
        (
            "slow moving",
            "Are there any cyclists going slowly {}?",
        ),  # up to 30 km/h
        (
            "fast",
            "Are there any cyclists going fast {}?",
        ),  # any faster than 30km/h
    ],
}

DIRECTION_STRINGS = {
    "towards": "towards the camera",
    "left": "towards the left of the camera",
    "right": "towards the right of the camera",
    "away": "away from the camera",
}

class FailureTypes(enum.IntEnum):
    UNKNOWN_OR_SIGN = 0
    NOT_VISIBLE = 1
    NO_MOVEMENT_DIR = 2
    OBJ_TYPE_NONE = 3
    TOO_FAR = 4
    TOO_FEW_PTS = 5
@register_prompt_generator
class ObjectBinaryPromptGenerator(BasePromptGenerator):
    """Generates binary (yes/no) questions about object presence."""

    def __init__(self) -> None:
        super().__init__()
        self.negative_sample_ratio = 0.5  # 50% negative samples

        self.prompt_functions = [
            self._object_facing_type,
            self._object_movement_direction_type,
        ]
        
        self.failure_counts = [0 for x in FailureTypes]

    def check_object(
        self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo
    ) -> bool:
        """Check object for distance, num_lidar_pts, visibility etc."""
        if obj.type in ["TYPE_UNKNOWN", "TYPE_SIGN"]:
            self.failure_counts[FailureTypes.UNKNOWN_OR_SIGN] += 1
            return False

        if not obj.is_visible_on_camera(frame, camera):
            self.failure_counts[FailureTypes.NOT_VISIBLE] += 1
            return False

        movement_dir_txt = obj.get_camera_movement_direction(camera, frame)
        if not movement_dir_txt:
            self.failure_counts[FailureTypes.NO_MOVEMENT_DIR] += 1
            return False

        obj_type = obj.type
        if obj_type is None:
            self.failure_counts[FailureTypes.OBJ_TYPE_NONE] += 1
            
            return False

        if np.linalg.norm(obj.get_centre()) > DISTANCE_BOUND:
            self.failure_counts[FailureTypes.TOO_FAR] += 1

            return False

        if getattr(obj, "num_lidar_points_in_box", 0.0) < MIN_LIDAR_PTS:
            self.failure_counts[FailureTypes.TOO_FEW_PTS] += 1

            return False
        
        # if obj.detection_difficulty_level == DifficultyLevelType.UNKNOWN:
        #     return False

        return True

    def _format_object_facing_type_question(
        self, obj_type: str, heading: str, movement_status: str
    ) -> str:
        """Format with specific grammar for each movement status, as the template doesn't always make sense."""
        for movement_cat, template in OBJECT_SPEED_CATS_QUESTIONS[obj_type]:
            if movement_cat == movement_status:
                return template.format(DIRECTION_STRINGS[heading])

        return f"Are there any {movement_status} {WAYMO_TYPE_MAPPING[obj_type].lower()}s facing {DIRECTION_STRINGS[heading]}?"

    def _format_object_movement_type_question(
        self, obj_type: str, movement_dir: str, movement_status: str
    ) -> str:
        """Format with specific grammar for each movement status, as the template doesn't always make sense."""
        for movement_cat, template in OBJECT_SPEED_CATS_QUESTIONS_MOVING[obj_type]:
            if movement_status == "stationary":
                return template
            if movement_cat == movement_status:
                return template.format(DIRECTION_STRINGS[movement_dir])

        if movement_status == "stationary":
            return f"Are there any stationary {WAYMO_TYPE_MAPPING[obj_type].lower()}s?"
        else:
            return f"Are there any {WAYMO_TYPE_MAPPING[obj_type].lower()}s that are {movement_status} and moving {DIRECTION_STRINGS[movement_dir]}?"

    def _object_movement_direction_type(self, camera: CameraInfo, frame: FrameInfo):
        """Generates binary questions about object type, movement status (moving or static) and movement direction."""
        direction_counts = defaultdict(int)

        for obj in frame.objects:
            if not self.check_object(obj, camera, frame):
                continue

            movement_status = obj.get_speed_category()
            movement_dir = obj.get_camera_movement_direction(camera, frame)
            print('obj.type, movement_dir.value, movement_status', obj.type, movement_dir.value, movement_status)
            key = (obj.type, movement_dir.value, movement_status)
            direction_counts[key] += 1

        questions = []
        answers = []

        # Add positive examples (where objects exist)
        for (
            obj_type,
            movement_dir_txt,
            movement_status,
        ), count in direction_counts.items():
            answer = "yes" if count > 0 else "no"
            questions.append(
                self._format_object_movement_type_question(
                    obj_type, movement_dir_txt, movement_status
                )
            )
            answers.append(answer)

        # Generate all possible combinations for negative sampling
        all_object_types = [
            x
            for x in WAYMO_TYPE_MAPPING
            if x is not None and (x not in ["TYPE_UNKNOWN", "TYPE_SIGN"])
        ]
        all_headings = [h.value for h in HeadingType]

        all_possible_combinations = []
        for obj_type in all_object_types:
            for heading in all_headings:
                for speed_cat, _, _ in OBJECT_SPEED_CATS[obj_type]:
                    all_possible_combinations.append((obj_type, heading, speed_cat))

        # Calculate how many negative samples we need
        negative_samples_needed = int(
            len(questions)
            * self.negative_sample_ratio
            / (1 - self.negative_sample_ratio)
        )

        # Use the existing function to generate negative samples
        positive_combinations = list(direction_counts.keys())
        negative_combinations = self._generate_negative_samples(
            positive_combinations, all_possible_combinations, negative_samples_needed
        )

        # # If we can't get enough negatives, reduce positives to maintain ratio
        # if len(negative_combinations) < negative_samples_needed:
        #     # Calculate how many positives we should keep
        #     max_positives = int(
        #         len(negative_combinations)
        #         * (1 - self.negative_sample_ratio)
        #         / self.negative_sample_ratio
        #     )
        #     if len(questions) > max_positives:
        #         # Randomly sample from positive examples
        #         positive_samples = list(zip(questions, answers))
        #         selected_positives = random.sample(positive_samples, max_positives)
        #         questions, answers = zip(*selected_positives)
        #         questions, answers = list(questions), list(answers)

        # Add the negative samples using the generated combinations
        for obj_type, movement_dir_txt, movement_status in negative_combinations:
            questions.append(
                self._format_object_movement_type_question(
                    obj_type, movement_dir_txt, movement_status
                )
            )
            answers.append("no")

        return questions, answers

    def _object_facing_type(self, camera, frame):
        """Generates binary questions about object type, movement status (moving or static) and heading (facing) combinations."""
        heading_counts = defaultdict(int)

        for obj in frame.objects:
            if not self.check_object(obj, camera, frame):
                continue

            heading_txt = obj.get_camera_heading_direction(frame, camera).value
            movement_status = obj.get_speed_category()
            key = (obj.type, heading_txt, movement_status)
            heading_counts[key] += 1

        questions = []
        answers = []

        # Add positive examples (where objects exist)
        for (obj_type, heading, movement_status), count in heading_counts.items():
            answer = "yes" if count > 0 else "no"
            questions.append(
                self._format_object_facing_type_question(
                    obj_type, heading, movement_status
                )
            )
            answers.append(answer)

        # Generate all possible combinations for negative sampling
        all_object_types = [
            x
            for x in WAYMO_TYPE_MAPPING
            if x is not None and (x not in ["TYPE_UNKNOWN", "TYPE_SIGN"])
        ]
        all_headings = [h.value for h in HeadingType]

        all_possible_combinations = []
        for obj_type in all_object_types:
            for heading in all_headings:
                for speed_cat, _, _ in OBJECT_SPEED_CATS[obj_type]:
                    all_possible_combinations.append((obj_type, heading, speed_cat))

        # Calculate how many negative samples we need
        negative_samples_needed = int(
            len(questions)
            * self.negative_sample_ratio
            / (1 - self.negative_sample_ratio)
        )

        # Use the existing function to generate negative samples
        positive_combinations = list(heading_counts.keys())
        negative_combinations = self._generate_negative_samples(
            positive_combinations, all_possible_combinations, negative_samples_needed
        )

        # # If we can't get enough negatives, reduce positives to maintain ratio
        # if len(negative_combinations) < negative_samples_needed:
        #     # Calculate how many positives we should keep
        #     max_positives = int(
        #         len(negative_combinations)
        #         * (1 - self.negative_sample_ratio)
        #         / self.negative_sample_ratio
        #     )
        #     if len(questions) > max_positives:
        #         # Randomly sample from positive examples
        #         positive_samples = list(zip(questions, answers))
        #         selected_positives = random.sample(positive_samples, max_positives)
        #         questions, answers = zip(*selected_positives)
        #         questions, answers = list(questions), list(answers)

        # Add the negative samples using the generated combinations
        for obj_type, heading, movement_status in negative_combinations:
            questions.append(
                self._format_object_facing_type_question(
                    obj_type, heading, movement_status
                )
            )
            answers.append("no")

        return questions, answers

    def _generate_negative_samples(self, positive_choices, all_choices, count_needed):
        """Generate negative samples from choices not in positive set."""
        negative_choices = [
            choice for choice in all_choices if choice not in positive_choices
        ]
        num_to_sample = min(count_needed, len(negative_choices))
        return (
            random.sample(negative_choices, num_to_sample) if negative_choices else []
        )

    def generate(
        self, frames
    ) -> List[Tuple[SingleImageMultipleChoiceQuestion, MultipleChoiceAnswer]]:
        """Generate binary (yes/no) questions."""
        samples = []

        frame = random.choice(frames)

        for camera in frame.cameras:
            # Generate questions using generators
            for question_func in self.prompt_functions:
                questions, answers = question_func(camera, frame)

                for question_text, answer_text in zip(questions, answers):
                    question = SingleImageMultipleChoiceQuestion(
                        image_path=camera.image_path,
                        question=question_text,
                        scene_id=frame.scene_id,
                        timestamp=frame.timestamp,
                        camera_name=camera.name,
                        generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                        question_name=question_func.__name__,
                        choices=["yes", "no"],
                    )

                    answer = MultipleChoiceAnswer(
                        choices=["yes", "no"], answer=answer_text
                    )
                    samples.append((question, answer))

        print('Failure Rates')
        for fail_type in FailureTypes:
            print(f'{fail_type.name}', self.failure_counts[fail_type.value])
        

        return samples

    def visualise_sample(
        self,
        question_obj: SingleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        pred_answer_obj: Optional[MultipleChoiceAnswer] = None,
        extra_text="",
        figsize=(12, 8),
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        import plotly.graph_objects as go
        from PIL import Image
        import numpy as np
        import textwrap
        from typing import Optional
        """
        Minimal Plotly visualization showing question, answer, and image.
        
        Args:
            question_obj: Question object with image_path and question text
            answer_obj: Answer object with answer text
            save_path: Path to save the visualization
            pred_answer_obj: Optional prediction answer object
            width: Figure width in pixels
            height: Figure height in pixels
        """
        
        # Load image
        try:
            print("question_obj.image_path", question_obj.image_path)
            img = Image.open(question_obj.image_path)
            # img_array = np.array(img)
            
            width = img.width
            height = img.height
        except Exception as e:
            # Create a placeholder if image can't be loaded
            img_array = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Create figure
        fig = go.Figure()
        
        # Add image
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=height,
                sizex=width,
                sizey=height,
                sizing="stretch",
                layer="below"
            )
        )
        
        # Prepare answer text and color
        answer_text = answer_obj.answer.lower()
        answer_color = "white"
        answer_symbol = ""
        
        if pred_answer_obj is not None:
            if pred_answer_obj.get_answer_text().lower() == answer_obj.answer.lower():
                answer_color = "green"
                answer_symbol = " ✓"
            else:
                answer_color = "red" 
                answer_symbol = " ✗"
        
        # Add question text at top
        question_text = textwrap.fill(question_obj.question, width=60)
        fig.add_annotation(
            text=f"Q: {question_text}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=34, color="black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="white",
            borderwidth=1
        )
        
        # Add ground truth answer at bottom
        fig.add_annotation(
            text=f"Answer: {answer_text.title()}",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=34, color="black", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="white",
            borderwidth=1
        )
        
        # Add prediction if available
        if pred_answer_obj is not None:
            pred_text = f"Prediction: {pred_answer_obj.get_answer_text().upper()}{answer_symbol}"
            fig.add_annotation(
                text=pred_text,
                xref="paper", yref="paper",
                x=0.02, y=0.10,
                xanchor="left", yanchor="bottom",
                showarrow=False,
                font=dict(size=34, color="white", family="Arial Black"),
                bgcolor=f"rgba({'0,128,0' if answer_color == 'green' else '255,0,0'},0.8)",
                bordercolor=answer_color,
                borderwidth=1
            )
        
        # Update layout for minimal appearance
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, width]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, height],
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        # Save the figure
        fig.write_image(save_path)
        
        return fig

    def visualise_sample_(
        self,
        question_obj: SingleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        pred_answer_obj: Optional[MultipleChoiceAnswer] = None,
        extra_text="",
        figsize=(12, 8),
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Simple visualization showing question, answer, and 2D bounding boxes for relevant objects.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: MultipleChoiceAnswer object with yes/no answer
            save_path: Path to save the visualization
            frames
            figsize: Figure size (width, height)
            text_fontsize: Font size for question/answer text
            title_fontsize: Font size for the title
            dpi: DPI for saved image
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np
        from pathlib import Path
        import textwrap

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Load and display the image
        try:
            img = Image.open(question_obj.image_path)
            img_array = np.array(img)
            ax.imshow(img_array)

        except Exception as e:
            # If image can't be loaded, show error
            ax.text(
                0.5,
                0.5,
                f"Image not found:\n{question_obj.image_path}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.axis("off")

        # Extract answer info
        answer_text = answer_obj.answer.lower()
        pred_answer_display = ""
        if pred_answer_obj is not None:
            if pred_answer_obj.get_answer_text().lower() == answer_obj.answer.lower():
                answer_symbol = "✓"
                answer_color = "green"
            else:
                answer_symbol = "✗"
                answer_color = "red"
                
            pred_answer_display = f"Prediction: {pred_answer_obj.get_answer_text().upper()} {answer_symbol}"

        # Add text overlay for question
        question_text = textwrap.fill(question_obj.question, width=60)
        ax.text(
            0.02,
            0.98,
            f"Q: {question_text}",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )

        # Add answer overlay
        answer_display = f"Answer: {answer_text.title()}" 
        ax.text(
            0.02,
            0.02,
            answer_display,
            transform=ax.transAxes,
            fontsize=text_fontsize + 2,
            weight="bold",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.8,
                edgecolor="white",
            ),
        )
        if pred_answer_display:
            ax.text(
                0.02,
                0.1,
                pred_answer_display,
                transform=ax.transAxes,
                fontsize=text_fontsize + 2,
                weight="bold",
                verticalalignment="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=answer_color,
                    alpha=0.8,
                    edgecolor=answer_color,
                ),
            )

        # Title
        camera_name = getattr(question_obj, "camera_name", "Unknown")
        plt.title(
            f"ObjectBinaryPromptGenerator - {camera_name}",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def get_question_type(self):
        return SingleImageMultipleChoiceQuestion

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer
