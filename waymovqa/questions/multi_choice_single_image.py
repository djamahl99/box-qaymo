from typing import Dict, List
from .base import BaseQuestion, QuestionType

import json

class MultiChoiceSingleImageQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_CHOICE_SINGLE_IMAGE
    image_path: str
    question: str
    choices: List[str]
    object_id: str
    scene_id: str
    timestamp: int
    camera_name: str

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            question_type=data['question_type'],
            image_path=data['image_path'], 
            question=data['question'],
            choices=data['choices'],
            object_id=data['object_id'],
            scene_id=data['scene_id'],
            timestamp=data['timestamp'],
            camera_name=data['camera_name']
        )