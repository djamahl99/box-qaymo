from typing import List, Dict
from .base import BaseQuestion, QuestionType


class MultipleImageMultipleChoiceQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_IMAGE
    image_paths: List[str]
    question: str
    choices: List[str]
    scene_id: str
    timestamp: int
    camera_names: List[str]
    generator_name: str

    def __hash__(self) -> int:
        return hash((self.question, self.scene_id, self.timestamp))

    def __eq__(self, other):
        if not isinstance(other, MultipleImageMultipleChoiceQuestion):
            return False
        return (
            self.question == other.question
            and self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
        )
