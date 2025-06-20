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
from io import BytesIO

from box_qaymo.answers.multiple_choice import MultipleChoiceAnswer
from box_qaymo.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from box_qaymo.data.scene_info import SceneInfo
from box_qaymo.data.object_info import (
    OBJECT_SPEED_CATS,
    HeadingType,
    MovementType,
    ObjectInfo,
)
from box_qaymo.data.frame_info import FrameInfo
from box_qaymo.data.camera_info import CameraInfo
from box_qaymo.data.laser_info import LaserInfo
from box_qaymo.prompt_generators.base import BasePromptGenerator
from box_qaymo.prompt_generators import register_prompt_generator

from box_qaymo.data.visualise import (
    draw_3d_wireframe_box_cv,
    generate_object_depth_buffer,
)
from box_qaymo.primitives import colors as labelled_colors
from box_qaymo.primitives import labels as finegrained_labels
from box_qaymo.primitives import CHOICES_OPTIONS

OBJECT_COLOR = (0, 0, 255)
OBJECT_COLOR_TEXT = "red"
OBJECT_RECT_THICKNESS = 6

DRAWN_BBOX_COLOR_QUESTIONS_CHOICES = [
    "What color is the object in the {} box? Choose from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
    "What color is the object highlighted in {}? Select from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
    "Which color best describes the object in the {} box? Options: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
]

LABEL_QUESTIONS_CHOICES = [
    "What type of object is in the {} box? Choose from: {{}}".format(OBJECT_COLOR_TEXT),
    "What is the object in the {} rectangle? Select from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
    "Which category best describes the object in the {} box? Options: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
]

HEADING_QUESTIONS_CHOICES = [
    "Which direction is the object in the {} box facing? Choose from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
    "What direction is the object highlighted in {} facing? Select from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
]

MOVEMENT_DIRECTION_QUESTIONS_CHOICES = [
    "Which direction is the object in the {} box moving? Choose from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
    "What direction is the object highlighted in {} moving? Select from: {{}}".format(
        OBJECT_COLOR_TEXT
    ),
]


@register_prompt_generator
class ObjectDrawnBoxPromptGenerator(BasePromptGenerator):
    """Generates questions about objects highlighted in te image."""

    def __init__(self) -> None:
        super().__init__()

        self.prompt_functions = [
            self._color_prompt,
            self._label_prompt,
            self._heading_prompt,
            self._speed_prompt,
            self._movement_direction_prompt,
        ]

    def _color_prompt(self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo):
        """Generate prompt based on the object color."""
        if obj.cvat_color is None:
            return None, None, None

        return labelled_colors, DRAWN_BBOX_COLOR_QUESTIONS_CHOICES, obj.cvat_color

    def _label_prompt(self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo):
        """Generate prompt based on the cvat label."""
        if obj.cvat_label is None:
            return None, None, None

        return finegrained_labels, LABEL_QUESTIONS_CHOICES, obj.cvat_label

    def _heading_prompt(self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo):
        """Generate prompt based on the object heading."""
        choices = [enum_val.value for enum_val in HeadingType]

        return (
            choices,
            HEADING_QUESTIONS_CHOICES,
            obj.get_camera_heading_direction(frame, camera),
        )

    def _movement_direction_prompt(
        self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo
    ):
        """Generate prompt based on the object heading."""
        choices = [enum_val.value for enum_val in MovementType]

        return (
            choices,
            MOVEMENT_DIRECTION_QUESTIONS_CHOICES,
            obj.get_camera_movement_direction(camera, frame),
        )

    def _speed_prompt(self, obj: ObjectInfo, camera: CameraInfo, frame: FrameInfo):
        speed = obj.get_speed()

        if speed is None:
            return None, None, None

        choices = []
        answer_txt = None
        for speed_cat, speed_lwbnd, speed_upbnd in OBJECT_SPEED_CATS[obj.type]:  # type: ignore
            choices.append(speed_cat)

            if (speed >= speed_lwbnd) and (speed <= speed_upbnd):
                answer_txt = speed_cat

        templates = [
            f"How fast would you describe the speed of the {obj.get_simple_type()}? "
            + "Choose from {}."
        ]

        return (
            choices,
            templates,
            answer_txt,
        )

    def generate(
        self, frames
    ) -> List[Tuple[SingleImageMultipleChoiceQuestion, MultipleChoiceAnswer]]:
        """Generates questions about objects with bounding boxes drawn on the frame."""
        samples = []

        # Track all objects across frames
        object_best_views = self._find_best_object_views(frames)

        for frame, camera, obj in object_best_views.values():
            if obj.type in ["TYPE_UNKNOWN", "TYPE_SIGN"]:
                continue

            if not obj.is_visible_on_camera(frame, camera):
                continue

            # img_vis = cv2.imread(camera.image_path)  # type: ignore

            # x1, y1, x2, y2 = obj.get_object_bbox_2d(frame, camera)

            # cv2.rectangle(
            #     img_vis, (x1, y1), (x2, y2), OBJECT_COLOR, OBJECT_RECT_THICKNESS  # type: ignore
            # )

            # _, buffer = cv2.imencode(".jpg", img_vis)  # or '.png'
            # img_base64 = base64.b64encode(buffer).decode("utf-8")

            for prompt_func in self.prompt_functions:
                prompt_out = prompt_func(obj, camera, frame)

                # e.g. if an object was not moving
                if any([x is None for x in prompt_out]):
                    continue

                choices, question_templates, answer_txt = prompt_out

                if len(choices) > CHOICES_OPTIONS:
                    # Remove correct answer from choices
                    distractors = [x for x in choices if x != answer_txt]

                    # Sample (CHOICES_OPTIONS - 1) distractors
                    sampled = np.random.choice(
                        distractors, CHOICES_OPTIONS - 1, replace=False
                    ).tolist()

                    # Add the correct answer back in
                    choices = sampled + [answer_txt]

                    # (Optional) Shuffle the final choices
                    np.random.shuffle(choices)

                if len(choices) < 2:
                    print(f"Received only {len(choices)} choices from {prompt_func}")
                    continue

                if answer_txt is None or answer_txt not in choices:
                    print(
                        f"answer_txt={answer_txt} is none or not in choices={choices}"
                    )
                    continue

                # Format options for all question texts
                formatted_options = self._get_formatted_options(choices)

                # Select one question template randomly
                question_template = random.choice(question_templates)

                # Create the question with formatted options
                question_text = question_template.format(formatted_options)

                x1, y1, x2, y2 = obj.get_object_bbox_2d(frame, camera)

                question = SingleImageMultipleChoiceQuestion(
                    # image_base64=img_base64,
                    image_path=camera.image_path,  # type: ignore
                    question=question_text,
                    choices=choices,
                    scene_id=obj.scene_id,
                    timestamp=frame.timestamp,
                    camera_name=camera.name,
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                    question_name=prompt_func.__name__,
                    data=dict(bbox=[x1, y1, x2, y2]),
                )

                answer = MultipleChoiceAnswer(
                    choices=choices,
                    answer=answer_txt,
                )

                samples.append((question, answer))

        return samples

    def visualise_sample_matplotlib(
        self,
        question_obj: SingleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        pred_answer_obj: Optional[MultipleChoiceAnswer] = None,
        extra_text="",
        figsize=(12, 8),
        box_color="green",
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Simple visualization showing question, choices over image with drawn object.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: MultipleChoiceAnswer
            save_path: Path to save the visualization
            frames:
            figsize: Figure size (width, height)
            box_color: Color for bounding boxes
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
        import base64
        from io import BytesIO

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Load and display the image
        try:
            # img = Image.open(BytesIO(base64.b64decode(question_obj.image_base64)))
            img = Image.open(question_obj.image_path)
            img_array = np.array(img)
            ax.imshow(img_array)

        except Exception as e:
            # If image can't be loaded, show error
            ax.text(
                0.5,
                0.5,
                f"Image could not be decoded: {str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.axis("off")

        # Get question text
        question_text = question_obj.question

        # Add text overlay for question
        question_text = textwrap.fill(question_text, width=60)
        ax.text(
            0.02,
            0.98,
            f"Q: {question_text}",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )
        
        if isinstance(question_obj.data, dict) and "bbox" in question_obj.data:
            x1, y1, x2, y2 = question_obj.data['bbox']
            width = x2 - x1
            height = y2 - y1
            
            ax.add_patch(patches.Rectangle(
                xy=(x1, y1), 
                width=width, 
                height=height, 
                fill=False,           # No fill (just border)
                edgecolor='red',      # Red border to match OpenCV (255,0,0)
                linewidth=3           # Thick line (OpenCV uses thickness=6)
            ))

        # Display choices and highlight the correct answer
        for idx, choice in enumerate(answer_obj.choices):
            # correct and pred
            if choice == answer_obj.answer and pred_answer_obj is not None and choice == pred_answer_obj.get_answer_text():
                choice_color = "lightgreen"
                choice_text = f"✓ {choice} (CORRECT)"
                text_color = "darkgreen"
            # correct
            elif choice == answer_obj.answer:
                choice_color = "lightgreen"
                choice_text = f"{choice}"
                text_color = "darkgreen"
            # incorrect and pred
            elif pred_answer_obj is not None and choice == pred_answer_obj.get_answer_text():
                choice_color = "lightcoral"
                choice_text = f"✗ {choice} (INCORRECT)"
                text_color = "darkred"
            # incorrect
            else:
                choice_color = "lightcoral"
                choice_text = f"{choice}"
                text_color = "darkred"

            ax.text(
                0.02,
                0.02 + idx * 0.06,  # Better spacing
                choice_text,
                transform=ax.transAxes,
                fontsize=text_fontsize,
                weight="bold",
                verticalalignment="bottom",
                color=text_color,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=choice_color,
                    alpha=0.8,
                    edgecolor=text_color,
                ),
            )

        # Title
        camera_name = getattr(question_obj, "camera_name", "Unknown")
        plt.title(
            f"Object Drawn Box QA - {camera_name}",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
    

    def visualise_sample(
        self,
        question_obj: SingleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        pred_answer_obj: Optional[MultipleChoiceAnswer] = None,
        extra_text="",
        figsize=(12, 8),
        box_color="green",
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Minimal Plotly visualization showing question, multiple choices, and image with bounding box.
        
        Args:
            question_obj: Question object with image_path, question text, and optional bbox data
            answer_obj: Answer object with choices and correct answer
            save_path: Path to save the visualization
            pred_answer_obj: Optional prediction answer object
        """
        import plotly.graph_objects as go
        from PIL import Image
        import textwrap
        from typing import Optional
        # Load image
        try:
            img = Image.open(question_obj.image_path)
            width, height = img.size
        except Exception as e:
            # Create a placeholder if image can't be loaded
            img = Image.new('RGB', (600, 400), color='white')
            width, height = img.size
        
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
        
        # Add bounding box if available
        if isinstance(question_obj.data, dict) and "bbox" in question_obj.data:
            x1, y1, x2, y2 = question_obj.data['bbox']
            fig.add_shape(
                type="rect",
                x0=x1, y0=height-y2,  # Flip y-coordinate for plotly
                x1=x2, y1=height-y1,
                line=dict(color="red", width=3),
                fillcolor="rgba(0,0,0,0)"  # No fill
            )
        
        # Add question text at top
        question_text = textwrap.fill(question_obj.question, width=60)
        fig.add_annotation(
            text=f"Q: {question_text}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=48, color="black", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="white",
            borderwidth=1
        )
        
        # Display choices
        pred_answer_text = pred_answer_obj.get_answer_text() if pred_answer_obj else None
        correct_answer = answer_obj.answer
        
        for idx, choice in enumerate(answer_obj.choices):
            # Determine styling based on correctness and prediction
            is_correct = (choice == correct_answer)
            is_predicted = (choice == pred_answer_text)
            
            if is_correct and is_predicted:
                # Correct prediction
                choice_text = f"✓ {choice}"
                bg_color = "rgba(0,128,0,0.8)"  # Green
                text_color = "white"
            elif is_correct and not is_predicted:
                # Correct but not predicted
                choice_text = f"{choice}"
                bg_color = "rgba(144,238,144,0.8)"  # Light green
                text_color = "darkgreen"
            elif not is_correct and is_predicted:
                # Incorrect prediction
                choice_text = f"✗ {choice}"
                bg_color = "rgba(255,0,0,0.8)"  # Red
                text_color = "white"
            else:
                # Not correct, not predicted
                choice_text = f"{choice}"
                bg_color = "rgba(240,128,128,0.8)"  # Light coral
                text_color = "darkred"
            
            # Add choice annotation
            fig.add_annotation(
                text=choice_text,
                xref="paper", yref="paper",
                x=0.02, y=0.02 + idx * 0.08,  # Stack choices from bottom
                xanchor="left", yanchor="bottom",
                showarrow=False,
                font=dict(size=48, color=text_color, family="Arial Black"),
                bgcolor=bg_color,
                bordercolor=text_color,
                borderwidth=2
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

    def get_metric_class(self) -> str:
        return "MultipleChoiceMetric"

    def get_answer_type(self):
        return MultipleChoiceAnswer

    def get_question_type(self):
        return SingleImageMultipleChoiceQuestion
