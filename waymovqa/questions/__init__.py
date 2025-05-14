import json
from typing import Dict

from .base import QuestionType, BaseQuestion
from .single_image import SingleImageQuestion
from .multi_image import MultipleImageQuestion
from .multi_prompt_single_image import MultiPromptSingleImageQuestion

QUESTION_TYPE_REGISTRY = {
    QuestionType.SINGLE_IMAGE: SingleImageQuestion,
    QuestionType.MULTI_IMAGE: MultipleImageQuestion,
    QuestionType.MULTI_PROMPT_SINGLE_IMAGE: MultiPromptSingleImageQuestion
}

def question_from_json(text: str) -> BaseQuestion:
    data = json.loads(text)
    cls = QUESTION_TYPE_REGISTRY[QuestionType(data["question_type"])]
    return cls.from_json(text)

def question_from_dict(data: Dict) -> BaseQuestion:
    cls = QUESTION_TYPE_REGISTRY[QuestionType(data["question_type"])]
    return cls.from_json(data)