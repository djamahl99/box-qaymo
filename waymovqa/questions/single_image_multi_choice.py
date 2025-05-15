from typing import Dict, List
from .base import BaseQuestion, QuestionType

import json

class SingleImageMultipleChoiceQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.SINGLE_IMAGE_MULTI_CHOICE
    image_path: str
    question: str
    choices: List[str]
    scene_id: str
    timestamp: int
    camera_name: str