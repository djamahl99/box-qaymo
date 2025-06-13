from .base import BaseAnswer
from .base import AnswerType
import json
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional

import numpy as np


class RawTextAnswer(BaseAnswer):
    """Typed format for raw text answer."""

    answer_type: AnswerType = AnswerType.RAW_TEXT
    text: str

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(text=data["text"])

    @classmethod
    def from_text(cls, text: str):
        return cls(text=text.strip())

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text)

        return cls(text=data["text"])

    def to_json(self):
        return json.dumps(dict(text=self.text, answer_type=str(self.answer_type)))

    def get_answer_text(self) -> str:
        return self.text
