from waymovqa.answers.object_2d import Object2DAnswer
from .base import BaseAnswer
from .base import AnswerType
import json
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional

import numpy as np


class MultiObject2DAnswer(BaseAnswer):
    """Container for multiple 2D object answers."""

    answer_type: AnswerType = AnswerType.MULTIPLE_OBJECT_2D
    boxes: List[List[float]]  # Format: List of [x1, y1, x2, y2]
    scores: List[float]
    prompt: Optional[str] = None

    @classmethod
    def from_single_object(cls, obj: Object2DAnswer, prompt: str = None):
        """Create a multi-object answer from a single object."""
        return cls(boxes=[obj.box], scores=[obj.score], prompt=prompt)

    @classmethod
    def from_text(cls, text: str):
        """Parse multiple object detection results from text."""
        data = json.loads(text)
        boxes = data["boxes"]
        scores = data.get("scores", [1.0] * len(boxes))
        prompt = data.get("prompt", None)
        return cls(boxes=boxes, scores=scores, prompt=prompt)

    def to_json(self):
        return json.dumps(
            dict(
                boxes=self.boxes,
                scores=self.scores,
                prompt=self.prompt,
                answer_type=str(self.answer_type),
            )
        )
