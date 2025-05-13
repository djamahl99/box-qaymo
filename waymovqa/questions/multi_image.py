from typing import List, Dict
from .base import BaseQuestion, QuestionType

class MultipleImageQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_IMAGE
    image_paths: List[str]
    question: str
    object_id: str
    scene_id: str
    timestamp: int
    camera_names: List[str]

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            question_type=data['question_type'],
            image_paths=data['image_paths'], 
            question=data['question'],
            object_id=data['object_id'],
            scene_id=data['scene_id'],
            timestamp=data['timestamp'],
            camera_names=data['camera_names']
        )