from typing import List, Dict
from .base import BaseQuestion, QuestionType


class MultipleImageQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_IMAGE
    image_paths: List[str]
    question: str
    scene_id: str
    timestamp: int
    camera_names: List[str]
    generator_name: str

    def __hash__(self) -> int:
        return hash((self.scene_id, self.timestamp, self.question))

    def __eq__(self, other):
        if not isinstance(other, MultipleImageQuestion):
            return False
        return (
            self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
            and self.question == other.question
        )
