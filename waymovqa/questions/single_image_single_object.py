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
    