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