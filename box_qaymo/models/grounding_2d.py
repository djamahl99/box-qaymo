from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    List,
    Union,
    Type,
    TypeVar,
    Generic,
    Optional,
    Tuple,
    Protocol,
)
from pathlib import Path
import json

from box_qaymo.models.base import BaseModel
from box_qaymo.questions.single_image_single_object import (
    SingleImageSingleObjectQuestion,
)
from box_qaymo.answers.object_2d import Object2DAnswer
from box_qaymo.metrics.coco import COCOMetric


class Grounding2DModel(ABC):
    @property
    def question_type(self):
        return SingleImageSingleObjectQuestion

    @property
    def answer_type(self):
        return Object2DAnswer

    @property
    def metric_type(self) -> Type[COCOMetric]:
        return COCOMetric
