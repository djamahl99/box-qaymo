from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional, Tuple, Protocol
from pathlib import Path
import json

from waymovqa.models.base import BaseModel
from waymovqa.questions.single_image import SingleImageQuestion
from waymovqa.answers.object_2d import Object2DAnswer
from waymovqa.metrics.coco import COCOMetric

class Grounding2DModel(ABC):
    @property
    @abstractmethod
    def question_type(self):
        return SingleImageQuestion

    @property
    @abstractmethod
    def answer_type(self):
        return Object2DAnswer

    @property
    @abstractmethod
    def metric_type(self) -> Type[COCOMetric]:
        return COCOMetric