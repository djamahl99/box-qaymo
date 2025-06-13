import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
import pytz

from box_qaymo.data import *
from box_qaymo.data.scene_info import SceneInfo
from box_qaymo.data.frame_info import FrameInfo, TimeOfDayType, WeatherType, LocationType
from box_qaymo.data.object_info import ObjectInfo
from box_qaymo.prompt_generators import BasePromptGenerator, register_prompt_generator

from box_qaymo.questions.multi_image_multi_choice import (
    MultipleImageMultipleChoiceQuestion,
)
from box_qaymo.questions.single_image import SingleImageQuestion
from box_qaymo.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from box_qaymo.questions.base import BaseQuestion
from box_qaymo.answers.multiple_choice import MultipleChoiceAnswer
from box_qaymo.metrics.multiple_choice import MultipleChoiceMetric
from box_qaymo.metrics.base import BaseMetric


class BaseSceneChoicePromptGenerator(BasePromptGenerator):
    """Generates questions about overall scene description."""

    QUESTION_CONFIGS = [
        (WeatherType, "weather", "How would you describe the weather out of {}?"),
        (
            TimeOfDayType,
            "time_of_day",
            "How would you describe the time of day out of {}?",
        ),
        (LocationType, "location", "Which city is this scene from out of {}?"),
    ]

    # Define more granular time periods
    DETAILED_TIME_PERIODS = [
        "Early Morning (5-8 AM)",
        "Morning (8-11 AM)",
        "Midday (11 AM-1 PM)",
        "Afternoon (1-5 PM)",
        "Evening (5-8 PM)",
        "Night (8 PM-5 AM)",
    ]

    def _convert_timestamp_to_time_period(
        self, timestamp_microseconds: int, location: str
    ) -> str:
        """
        Convert timestamp in microseconds to a specific time period,
        using the appropriate timezone based on location.
        """
        # Map locations to timezone strings
        location_to_timezone = {
            "location_sf": "America/Los_Angeles",  # San Francisco
            "location_phx": "America/Phoenix",  # Phoenix (no DST)
            "location_mtv": "America/Los_Angeles",  # Mountain View
            "location_la": "America/Los_Angeles",  # Los Angeles
            "location_det": "America/Detroit",  # Detroit
            "location_sea": "America/Los_Angeles",  # Seattle
            "location_chd": "America/Phoenix",  # Chandler, AZ
            "location_other": "America/Los_Angeles",  # Default
        }

        # Get location from frame
        timezone_str = location_to_timezone.get(location, "America/Los_Angeles")

        # Convert the timestamp using the appropriate timezone
        timestamp_seconds = timestamp_microseconds / 1_000_000
        utc_dt = datetime.fromtimestamp(timestamp_seconds, tz=pytz.UTC)
        local_tz = pytz.timezone(timezone_str)
        local_dt = utc_dt.astimezone(local_tz)

        hour = local_dt.hour

        # Determine the time period
        if 5 <= hour < 8:
            return "Early Morning (5-8 AM)"
        elif 8 <= hour < 11:
            return "Morning (8-11 AM)"
        elif 11 <= hour < 13:
            return "Midday (11 AM-1 PM)"
        elif 13 <= hour < 17:
            return "Afternoon (1-5 PM)"
        elif 17 <= hour < 20:
            return "Evening (5-8 PM)"
        else:
            return "Night (8 PM-5 AM)"

    def _generate_prompts(self, frame: FrameInfo):
        """Generate scene description questions."""
        assert frame is not None  # really need the frame for weather/time_of_day

        samples = []

        for enum_type, attr_name, question_template in self.QUESTION_CONFIGS:
            # Get options based on the enum type
            if enum_type == LocationType:
                # For LocationType, use enum keys (city names)
                options = [
                    member.name.replace("_", " ").title() for member in enum_type
                ]

                # Map the internal enum value to the display name for the answer
                answer_value = getattr(frame, attr_name)
                # Find the enum member with this value and get its name
                answer = next(
                    (
                        member.name.replace("_", " ").title()
                        for member in enum_type
                        if member.value == answer_value
                    ),
                    None,
                )
            else:
                # For other enum types, continue using values as before
                options = [enum_val.value for enum_val in enum_type]
                answer = getattr(frame, attr_name)

            # Check if time_of_day / weather / location was not parsed correctly
            if answer is None:
                continue

            # Format options for the question text
            formatted_options = ", ".join(options[:-1]) + " or " + options[-1]

            # Create the question with formatted options
            question = question_template.format(formatted_options)

            # Create and append the question-answer pair
            samples.append(
                (
                    dict(
                        image_paths=[cam.image_path for cam in frame.cameras],
                        question=question,
                        camera_names=[cam.name for cam in frame.cameras],
                        scene_id=frame.scene_id,
                        timestamp=frame.timestamp,
                    ),
                    dict(choices=options, answer=answer),
                )
            )

        # Add the more detailed time-of-day question using timestamp
        detailed_time_period = self._convert_timestamp_to_time_period(
            frame.timestamp, frame.location if frame.location else LocationType.OTHER
        )

        # Create the detailed time period question
        samples.append(
            (
                dict(
                    image_paths=[cam.image_path for cam in frame.cameras],
                    question="What time of day does this scene appear to be from out of: "
                    + ", ".join(self.DETAILED_TIME_PERIODS[:-1])
                    + " or "
                    + self.DETAILED_TIME_PERIODS[-1]
                    + "?",
                    scene_id=frame.scene_id,
                    timestamp=frame.timestamp,
                    camera_names=[cam.name for cam in frame.cameras],
                ),
                dict(choices=self.DETAILED_TIME_PERIODS, answer=detailed_time_period),
            )
        )

        return samples

    def get_question_type(self):
        return BaseQuestion

    def get_answer_type(self):
        return MultipleChoiceAnswer

    def get_metric_class(self):
        return MultipleChoiceMetric


# @register_prompt_generator
class SceneMultipleImageMultiChoicePromptGenerator(BaseSceneChoicePromptGenerator):
    """Generates multi-image questions."""

    def generate(self, frames):
        # choose a random frame
        frame = random.choice(frames)

        # Generate for one frame (given that )
        samples = self._generate_prompts(frame)

        samples_parsed = []
        for question, answer in samples:
            samples_parsed.append(
                (
                    MultipleImageMultipleChoiceQuestion(
                        image_paths=question["image_paths"],
                        question=question["question"],
                        choices=answer["choices"],
                        camera_names=question["camera_names"],
                        scene_id=question["scene_id"],
                        timestamp=question["timestamp"],
                        generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                        question_name="scene_question",
                    ),
                    MultipleChoiceAnswer(
                        choices=answer["choices"], answer=answer["answer"]
                    ),
                )
            )

        return samples_parsed

    def get_question_type(self):
        return MultipleImageMultipleChoiceQuestion

    def visualise_sample(
        self,
        question_obj: SingleImageQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
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

        idx = question_obj.camera_names.index("FRONT")

        image_path = question_obj.image_paths[idx]
        camera_name = question_obj.camera_names[idx]

        # Load and display the image
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            ax.imshow(img_array)

        except Exception as e:
            # If image can't be loaded, show error
            ax.text(
                0.5,
                0.5,
                f"Image not found:\n{image_path}",
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

        # Set colors based on answer
        if answer_text == "yes":
            answer_color = "green"
            answer_symbol = "✓"
        else:
            answer_color = "red"
            answer_symbol = "✗"

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
        answer_display = f"A: {answer_text.upper()} {answer_symbol}"
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
                facecolor=answer_color,
                alpha=0.8,
                edgecolor=answer_color,
            ),
        )

        # Title
        plt.title(
            f"Scene Single Image Multi Choice - {camera_name}",
            fontsize=title_fontsize,
            pad=20,
        )

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()


# @register_prompt_generator
class SceneSingleImageMultiChoicePromptGenerator(BaseSceneChoicePromptGenerator):
    """Generates single-image questions."""

    def generate(self, frames):
        # choose a random frame
        frame = random.choice(frames)

        # Generate for one frame (given that )
        samples = self._generate_prompts(frame)

        samples_parsed = []
        for question, answer in samples:
            image_paths, camera_names = (
                question["image_paths"],
                question["camera_names"],
            )

            # TODO: change from constant FRONT
            idx = camera_names.index("FRONT")

            image_path = image_paths[idx]
            camera_name = camera_names[idx]

            samples_parsed.append(
                (
                    SingleImageMultipleChoiceQuestion(
                        image_path=image_path,
                        question=question["question"],
                        choices=answer["choices"],
                        camera_name=camera_name,
                        scene_id=question["scene_id"],
                        timestamp=question["timestamp"],
                        generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                        question_name="scene_question",
                    ),
                    MultipleChoiceAnswer(
                        choices=answer["choices"], answer=answer["answer"]
                    ),
                )
            )

        return samples_parsed

    def visualise_sample(
        self,
        question_obj: SingleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
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

        # Set colors based on answer
        if answer_text == "yes":
            answer_color = "green"
            answer_symbol = "✓"
        else:
            answer_color = "red"
            answer_symbol = "✗"

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
        answer_display = f"A: {answer_text.upper()} {answer_symbol}"
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
                facecolor=answer_color,
                alpha=0.8,
                edgecolor=answer_color,
            ),
        )

        # Title
        camera_name = getattr(question_obj, "camera_name", "Unknown")
        plt.title(
            f"Scene Single Image Multi Choice - {camera_name}",
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
