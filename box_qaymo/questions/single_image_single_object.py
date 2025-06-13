from typing import Dict
from .base import BaseQuestion, QuestionType


class SingleImageSingleObjectQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.SINGLE_IMAGE
    image_path: str
    question: str
    object_id: str
    scene_id: str
    timestamp: int
    camera_name: str
    generator_name: str

    def __hash__(self) -> int:
        return hash(
            (
                self.question,
                self.scene_id,
                self.timestamp,
                self.object_id,
                self.camera_name,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, SingleImageSingleObjectQuestion):
            return False
        return (
            self.question == other.question
            and self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
            and self.object_id == other.object_id
            and self.camera_name == other.camera_name
        )
