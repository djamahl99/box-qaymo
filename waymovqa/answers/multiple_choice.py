from .base import BaseAnswer
from .base import AnswerType
import json
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional


class MultipleChoiceAnswer(BaseAnswer):
    """Typed format for raw text answer."""

    answer_type: AnswerType = AnswerType.MULTIPLE_CHOICE
    choices: List[str]
    answer: str

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text)

        return cls(choices=data['choices'], answer=data['answer'])
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(choices=data['choices'], answer=data['answer'])

    def to_json(self):
        return json.dumps(dict(choices=self.choices, answer=self.answer, answer_type=str(self.answer_type)))
