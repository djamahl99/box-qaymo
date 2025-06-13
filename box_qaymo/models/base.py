from abc import ABC, abstractmethod
from typing import Type
from box_qaymo.questions.base import BaseQuestion
from box_qaymo.answers.base import BaseAnswer
from box_qaymo.metrics.base import BaseMetric


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
