from typing import List, Dict
from .base import BaseQuestion, QuestionType


class MultipleImageSingleObject(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_IMAGE_SINGLE_OBJECT
    image_paths: List[str]
    question: str
    object_id: str
    scene_id: str
    timestamp: int
    camera_names: List[str]
    generator_name: str

    def __hash__(self) -> int:
        return hash((self.question, self.scene_id, self.timestamp, self.object_id))

    def __eq__(self, other):
        if not isinstance(other, MultipleImageSingleObject):
            return False
        return (
            self.question == other.question
            and self.scene_id == other.scene_id
            and self.timestamp == other.timestamp
            and self.object_id == other.object_id
        )
