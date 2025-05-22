from typing import Dict, List
from .base import BaseQuestion, QuestionType

import json


class SingleImageMultipleChoiceQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.SINGLE_IMAGE_MULTI_CHOICE
    image_path: str
    question: str
    choices: List[str]
    scene_id: str
    timestamp: int
    camera_name: str
    generator_name: str

    def __hash__(self) -> int:
        return hash((self.scene_id, self.timestamp, self.camera_name, self.question))

    def __eq__(self, other):
        if not isinstance(other, SingleImageMultipleChoiceQuestion):
            return False
        return (
            self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
            and self.camera_name == other.camera_name
            and self.question == other.question
        )


class SingleBase64ImageMultipleChoiceQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.SINGLE_IMAGE
    image_base64: str
    question: str
    choices: List[str]
    scene_id: str
    timestamp: int
    camera_name: str
    generator_name: str

    def __hash__(self) -> int:
        return hash((self.scene_id, self.timestamp, self.camera_name, self.question))

    def __eq__(self, other):
        if not isinstance(other, SingleBase64ImageMultipleChoiceQuestion):
            return False
        return (
            self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
            and self.camera_name == other.camera_name
            and self.question == other.question
        )
