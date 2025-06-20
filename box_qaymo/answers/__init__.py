import json
from typing import Dict

from .base import AnswerType, BaseAnswer
from .object_2d import Object2DAnswer
from .multi_object_2d import MultiObject2DAnswer
from .raw_text import RawTextAnswer
from .multiple_choice import MultipleChoiceAnswer
from .regression import RegressionAnswer

ANSWER_TYPE_REGISTRY = {
    AnswerType.OBJECT_2D: Object2DAnswer,
    AnswerType.MULTIPLE_OBJECT_2D: MultiObject2DAnswer,
    AnswerType.MULTIPLE_CHOICE: MultipleChoiceAnswer,
    AnswerType.RAW_TEXT: RawTextAnswer,
    AnswerType.REGRESSION: RegressionAnswer,
}


def answer_from_json(text: str) -> BaseAnswer:
    data = json.loads(text)
    cls = ANSWER_TYPE_REGISTRY[AnswerType(data["answer_type"])]
    return cls.from_json(text)


def answer_from_dict(data: Dict) -> BaseAnswer:
    cls = ANSWER_TYPE_REGISTRY[AnswerType(data["answer_type"])]
    return cls.from_json(data)
