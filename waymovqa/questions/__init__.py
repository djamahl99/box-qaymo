import json
from typing import Dict

from .base import QuestionType, BaseQuestion
from .single_image_single_object import SingleImageSingleObjectQuestion
from .single_image_multi_prompt import SingleImageMultiplePromptQuestion
from .single_image_multi_choice import (SingleImageMultipleChoiceQuestion, SingleBase64ImageMultipleChoiceQuestion)
from .multi_image import MultipleImageQuestion
from .multi_image_single_object import MultipleImageSingleObject

QUESTION_TYPE_REGISTRY = {
    QuestionType.SINGLE_IMAGE: SingleImageSingleObjectQuestion,
    QuestionType.SINGLE_IMAGE_MULTI_PROMPT: SingleImageMultiplePromptQuestion,
    QuestionType.SINGLE_IMAGE_MULTI_CHOICE: SingleImageMultipleChoiceQuestion,
    QuestionType.SINGLE_BASE64_IMAGE_MULTI_CHOICE: SingleBase64ImageMultipleChoiceQuestion,
    # QuestionType.MULTI_IMAGE_LIDAR: ,

    QuestionType.MULTI_IMAGE: MultipleImageQuestion,
    QuestionType.MULTI_IMAGE_SINGLE_OBJECT: MultipleImageSingleObject
}

def question_from_json(text: str) -> BaseQuestion:
    data = json.loads(text)
    cls = QUESTION_TYPE_REGISTRY[QuestionType(data["question_type"])]
    return cls.from_json(text)

def question_from_dict(data: Dict) -> BaseQuestion:
    cls = QUESTION_TYPE_REGISTRY[QuestionType(data["question_type"])]
    return cls.from_json(data)