from typing import List, Dict
from .base import BaseQuestion, QuestionType


class SingleImageQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.SINGLE_IMAGE
    image_path: str
    camera_name: str
    question: str
    scene_id: str
    timestamp: int
    generator_name: str

    def __hash__(self) -> int:
        return hash((self.question, self.scene_id, self.timestamp, self.camera_name))

    def __eq__(self, other):
        if not isinstance(other, SingleImageQuestion):
            return False
        return (
            self.question == other.question
            and self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
            and self.camera_name == other.camera_name
        )
