from .base import BaseAnswer
from .base import AnswerType
import json
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional

import numpy as np


class RegressionAnswer(BaseAnswer):
    """Typed format for raw text answer."""

    answer_type: AnswerType = AnswerType.REGRESSION
    value: float

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(value=data["value"])

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text)

        return cls(value=data["value"])

    def to_json(self):
        return json.dumps(dict(value=self.value, answer_type=str(self.answer_type)))

    def get_answer_text(self) -> str:
        return f"{self.value:.2f}"
