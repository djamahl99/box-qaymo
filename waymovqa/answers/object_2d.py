from .base import BaseAnswer
from .base import AnswerType
import json
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional

import numpy as np


class Object2DAnswer(BaseAnswer):
    """Typed format for 2D Grounding answers."""

    answer_type: AnswerType = AnswerType.OBJECT_2D
    box: List[float]  # Format: [x1, y1, x2, y2]
    score: float
    object_id: str
    prompt: Optional[str] = None

    @classmethod
    def from_text(cls, text: str):
        """Parse an object detection result from text."""
        data = json.loads(text)
        box = data["box"]
        score = data.get("score", 1.0)
        prompt = data.get("prompt", None)
        object_id = data.get("object_id", None)
        return cls(box=box, score=score, prompt=prompt)

    @classmethod
    def from_json(cls, text: str):
        """Parse an object detection result from text."""
        data = json.loads(text)
        box = data["box"]
        score = data.get("score", 1.0)
        prompt = data.get("prompt", None)
        object_id = data.get("object_id", None)
        return cls(box=box, score=score, prompt=prompt)

    @classmethod
    def from_dict(cls, data: Dict):
        """Parse an object detection result from dict."""
        box = data["box"]
        score = data.get("score", 1.0)
        object_id = data.get("object_id", None)
        return cls(box=box, score=score, object_id=object_id)

    def to_json(self):
        return json.dumps(
            dict(
                box=self.box,
                score=self.score,
                prompt=self.prompt,
                object_id=self.object_id,
                answer_type=str(self.answer_type),
            )
        )
