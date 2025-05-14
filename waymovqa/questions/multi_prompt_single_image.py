from typing import List
from pydantic import BaseModel
from .base import BaseQuestion, QuestionType

from waymovqa.answers.base import BaseAnswer

class PromptEntry(BaseModel):
    prompt: str
    object_ids: List[str]  # Object(s) relevant to this prompt
    answers: List[BaseAnswer] 

class MultiPromptSingleImageQuestion(BaseQuestion):
    question_type: QuestionType = QuestionType.MULTI_PROMPT_SINGLE_IMAGE
    image_path: str
    scene_id: str
    timestamp: float
    camera_name: str
    prompts: List["PromptEntry"]