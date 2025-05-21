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
