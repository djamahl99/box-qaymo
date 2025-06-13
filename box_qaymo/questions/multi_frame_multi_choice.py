from typing import List, Dict
from .base import BaseQuestion, QuestionType


class MultipleFrameMultipleChoiceQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_FRAME_MULTI_CHOICE
    current_image_path: str
    previous_image_path: str
    
    current_timestamp: int
    previous_timestamp: int
    
    question: str
    choices: List[str]
    scene_id: str
    camera_name: str
    generator_name: str
    
    current_bbox: List[int]
    previous_bbox: List[int]

    def __hash__(self) -> int:
        return hash((self.question, self.scene_id, self.current_timestamp))

    def __eq__(self, other):
        if not isinstance(other, MultipleFrameMultipleChoiceQuestion):
            return False
        return (
            self.question == other.question
            and self.scene_id == other.scene_id
            and self.current_timestamp == other.current_timestamp
        )
