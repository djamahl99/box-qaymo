import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Set, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict

from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.data.visualise import draw_3d_wireframe_box_cv
from waymovqa.metrics.coco import COCOMetric
from waymovqa.prompt_generators.base import BasePromptGenerator
from waymovqa.prompt_generators import register_prompt_generator
from waymovqa.answers import Object2DAnswer, MultiObject2DAnswer, BaseAnswer
from waymovqa.questions import (
    SingleImageSingleObjectQuestion,
    SingleImageMultiplePromptQuestion,
)
from waymovqa.questions.single_image_multi_prompt import PromptEntry

MIN_OBJECT_SIZE = 32


# @register_prompt_generator # avoid use by default
class Grounding2DPromptGenerator(BasePromptGenerator):
    """Generates questions about object locations with multiple prompts per image."""

    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        frames,
    ) -> List[Tuple[SingleImageMultiplePromptQuestion, BaseAnswer]]:
        samples = []

        # Track all objects across frames to find good instances
        object_best_views = self._find_best_object_views(frames)

        # Map timestamp to frames list index
        timestamp_to_frame = {frame.timestamp: frame for frame in frames}

        frame_cams = {}
        frame_cam_set = set()
        for frame, camera, obj in object_best_views.values():
            key = (frame.timestamp, camera.name)
            frame_cam_set.add(key)

            frame_cams.setdefault(key, {})
            frame_cams[key] = camera

        # Generate questions for each object using its best frame/camera
        for frame_timestamp, camera_name in frame_cam_set:
            frame = timestamp_to_frame[frame_timestamp]
            camera = frame_cams[frame_timestamp, camera_name]

            prompt_to_objs = defaultdict(list)

            for obj in frame.objects:
                # Build prompt string
                prompt = obj.get_object_description()

                if prompt is None:
                    continue

                prompt_to_objs[prompt].append(obj)

            prompt_entries = []
            for prompt, objs in prompt_to_objs.items():
                boxes, object_ids = [], []

                for obj in objs:
                    if not obj.is_visible_on_camera(frame, camera):
                        continue
                    uvdok = obj.project_to_image(
                        frame_info=frame, camera_info=camera, return_depth=True
                    )
                    u, v, depth, ok = uvdok.transpose()
                    ok = ok.astype(bool)

                    x_min, x_max = int(min(u)), int(max(u))
                    y_min, y_max = int(min(v)), int(max(v))

                    x_min, x_max = [
                        max(min(x, camera.width), 0) for x in [x_min, x_max]
                    ]
                    y_min, y_max = [
                        max(min(y, camera.height), 0) for y in [y_min, y_max]
                    ]

                    boxes.append([x_min, y_min, x_max, y_max])
                    object_ids.append(obj.id)

                if boxes and len(boxes) > 0:
                    prompt_entry = PromptEntry(
                        prompt=prompt,
                        object_ids=object_ids,
                        answers=[
                            Object2DAnswer(
                                box=box, score=1.0, prompt=prompt, object_id=obj_id
                            )
                            for box, obj_id in zip(boxes, object_ids)
                        ],
                    )
                    prompt_entries.append(prompt_entry)

            if not prompt_entries:
                continue

            multi_prompt_question = SingleImageMultiplePromptQuestion(
                image_path=camera.image_path,
                scene_id=frame.scene_id,
                timestamp=frame.timestamp,
                camera_name=camera.name,
                prompts=prompt_entries,
                generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
            )

            samples.append((multi_prompt_question, None))

        return samples

    def visualise_sample(
        self,
        question_obj: SingleImageMultiplePromptQuestion,
        answer_obj: None,
        save_path,
        frames,
        figsize=(12, 8),
        box_color="green",
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Simple visualization showing question, answer, and 2D bounding boxes for relevant objects.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: not used
            save_path: Path to save the visualization
            frames
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

        # choose the first prompt
        import pprint

        pprint.pp(question_obj)
        prompt_entry = question_obj.prompts[0]
        question_text = prompt_entry.prompt

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

        # Draw bounding boxes for answers
        for answer in prompt_entry.answers:
            x1, y1, x2, y2 = answer.box
            width = x2 - x1  # Fixed: was x2 - x2
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1),  # Fixed: was (x1, y2)
                width,
                height,
                linewidth=3,
                edgecolor=box_color,
                facecolor="none",
            )
            ax.add_patch(rect)

        # Title
        camera_name = getattr(question_obj, "camera_name", "Unknown")
        plt.title(
            f"Grounding2DPromptGenerator - {camera_name} - {len(prompt_entry.answers)} boxes",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def get_question_type(self):
        return SingleImageMultiplePromptQuestion

    def get_answer_type(self):
        return Union[Object2DAnswer, MultiObject2DAnswer]

    def get_metric_class(self):
        return COCOMetric
