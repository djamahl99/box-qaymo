from typing import Dict
from .base import BaseQuestion, QuestionType

class SingleImageQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.SINGLE_IMAGE
    image_path: str
    question: str
    object_id: str
    scene_id: str
    timestamp: int

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            question_type=data['question_type'],
            image_path=data['image_path'], 
            question=data['question'],
            object_id=data['object_id'],
            scene_id=data['scene_id'],
            timestamp=data['timestamp'],
        )