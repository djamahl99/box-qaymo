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