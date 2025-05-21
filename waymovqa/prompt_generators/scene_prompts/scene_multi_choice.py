import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
import pytz

from waymovqa.data import *
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.frame_info import FrameInfo, TimeOfDayType, WeatherType, LocationType
from waymovqa.data.object_info import ObjectInfo
from waymovqa.prompt_generators import BasePromptGenerator, register_prompt_generator

from waymovqa.questions.multi_image_multi_choice import MultipleImageMultipleChoiceQuestion
from waymovqa.questions.single_image_multi_choice import SingleImageMultipleChoiceQuestion
from waymovqa.questions.base import BaseQuestion
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.metrics.multiple_choice import MultipleChoiceMetric
from waymovqa.metrics.base import BaseMetric

class BaseSceneChoicePromptGenerator(BasePromptGenerator):
    """Generates questions about overall scene description."""

    QUESTION_CONFIGS = [
        (WeatherType, "weather", "How would you describe the weather out of {}?"),
        (TimeOfDayType, "time_of_day", "How would you describe the time of day out of {}?"),
        (LocationType, "location", "Which city is this scene from out of {}?")
    ]

    # Define more granular time periods
    DETAILED_TIME_PERIODS = [
        "Early Morning (5-8 AM)", 
        "Morning (8-11 AM)", 
        "Midday (11 AM-1 PM)", 
        "Afternoon (1-5 PM)", 
        "Evening (5-8 PM)", 
        "Night (8 PM-5 AM)"
    ]
    
    def _convert_timestamp_to_time_period(self, timestamp_microseconds: int, location: str) -> str:
        """
        Convert timestamp in microseconds to a specific time period,
        using the appropriate timezone based on location.
        """
        # Map locations to timezone strings
        location_to_timezone = {
            'location_sf': 'America/Los_Angeles',  # San Francisco
            'location_phx': 'America/Phoenix',     # Phoenix (no DST)
            'location_mtv': 'America/Los_Angeles', # Mountain View
            'location_la': 'America/Los_Angeles',  # Los Angeles
            'location_det': 'America/Detroit',     # Detroit
            'location_sea': 'America/Los_Angeles', # Seattle
            'location_chd': 'America/Phoenix',     # Chandler, AZ
            'location_other': 'America/Los_Angeles', # Default
        }
        
        # Get location from frame
        timezone_str = location_to_timezone.get(location, 'America/Los_Angeles')
        
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

    def _generate_prompts(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ):
        """Generate scene description questions."""
        assert frame is not None # really need the frame for weather/time_of_day
        
        samples = []

        for enum_type, attr_name, question_template in self.QUESTION_CONFIGS:
            # Get options based on the enum type
            if enum_type == LocationType:
                # For LocationType, use enum keys (city names)
                options = [member.name.replace('_', ' ').title() for member in enum_type]
                
                # Map the internal enum value to the display name for the answer
                answer_value = getattr(frame, attr_name)
                # Find the enum member with this value and get its name
                answer = next((member.name.replace('_', ' ').title() 
                            for member in enum_type 
                            if member.value == answer_value), None)
            else:
                # For other enum types, continue using values as before
                options = [enum_val.value for enum_val in enum_type]
                answer = getattr(frame, attr_name)
            
            # Check if time_of_day / weather / location was not parsed correctly
            if answer is None:
                continue
            
            # Format options for the question text
            formatted_options = ', '.join(options[:-1]) + ' or ' + options[-1]
            
            # Create the question with formatted options
            question = question_template.format(formatted_options)
            
            # Create and append the question-answer pair
            samples.append((
                dict(
                    image_paths=[cam.image_path for cam in frame.cameras],
                    question=question,
                    camera_names=[cam.name for cam in frame.cameras],
                    scene_id=frame.scene_id,
                    timestamp=frame.timestamp
                ),
                dict(
                    choices=options,
                    answer=answer
                )
            ))

        # Add the more detailed time-of-day question using timestamp
        detailed_time_period = self._convert_timestamp_to_time_period(frame.timestamp, frame.location)
        
        # Create the detailed time period question
        samples.append((
            dict(
                image_paths=[cam.image_path for cam in frame.cameras],
                question="What time of day does this scene appear to be from out of: " + 
                         ', '.join(self.DETAILED_TIME_PERIODS[:-1]) + ' or ' + self.DETAILED_TIME_PERIODS[-1] + "?",
                scene_id=frame.scene_id,
                timestamp=frame.timestamp,
                camera_names=[cam.name for cam in frame.cameras],

            ),
            dict(
                choices=self.DETAILED_TIME_PERIODS,
                answer=detailed_time_period
            )
        ))
        
        return samples
    
    def get_question_type(self):
        return BaseQuestion

    def get_answer_type(self):
        return MultipleChoiceAnswer

    def get_metric_class(self) -> BaseMetric:
        return MultipleChoiceMetric

    def get_supported_methods(self):
        return ['scene']

@register_prompt_generator
class SceneMultipleImageMultiChoicePromptGenerator(BaseSceneChoicePromptGenerator):
    """Generates multi-image questions."""
    
    def generate(self, scene, objects, frame, frames):
        # Multi-image specific implementation
        samples = self._generate_prompts(scene, objects, frame)

        samples_parsed = []
        for question, answer in samples:
            samples_parsed.append((
                MultipleImageMultipleChoiceQuestion(
                    image_paths=question['image_paths'],
                    question=question['question'],
                    choices=answer['choices'],
                    camera_names=question['camera_names'],
                    scene_id=question['scene_id'],
                    timestamp=question['timestamp'],
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                ),
                MultipleChoiceAnswer(
                    choices=answer['choices'],
                    answer=answer['answer']
                )
            ))

        return samples_parsed

    def get_question_type(self):
        return MultipleImageMultipleChoiceQuestion

@register_prompt_generator
class SceneSingleImageMultiChoicePromptGenerator(BaseSceneChoicePromptGenerator):
    """Generates single-image questions."""
    
    def generate(self, scene, objects, frame, frames):
        # Multi-image specific implementation
        samples = self._generate_prompts(scene, objects, frame)

        samples_parsed = []
        for question, answer in samples:
            image_paths, camera_names = question['image_paths'], question['camera_names']

            # TODO: change from constant FRONT
            idx = camera_names.index('FRONT')

            image_path = image_paths[idx]
            camera_name = camera_names[idx]

            samples_parsed.append((
                SingleImageMultipleChoiceQuestion(
                    image_path=image_path,
                    question=question['question'],
                    choices=answer['choices'],
                    camera_name=camera_name,
                    scene_id=question['scene_id'],
                    timestamp=question['timestamp'],
                        generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                ),
                MultipleChoiceAnswer(
                    choices=answer['choices'],
                    answer=answer['answer']
                )
            ))

        return samples_parsed
    
    def get_question_type(self):
        return SingleImageMultipleChoiceQuestion