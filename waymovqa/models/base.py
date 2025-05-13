from abc import ABC, abstractmethod
from typing import Type
from waymovqa.questions.base import BaseQuestion
from waymovqa.answers.base import BaseAnswer
from waymovqa.metrics.base import BaseMetric

class BaseModel(ABC):
    @property
    @abstractmethod
    def question_type(self) -> Type[BaseQuestion]:
        pass

    @property
    @abstractmethod
    def answer_type(self) -> Type[BaseAnswer]:
        pass

    @property
    @abstractmethod
    def metric_type(self) -> Type[BaseMetric]:
        pass