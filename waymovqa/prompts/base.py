from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional, Tuple

from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.answers import BaseAnswer
from waymovqa.questions import BaseQuestion

# Update BasePromptGenerator with typing information
TQ = TypeVar("T", bound=BaseAnswer)
TA = TypeVar("T", bound=BaseQuestion)


class BasePromptGenerator(Generic[T], ABC):
    """Base class for all prompt generators with typed answers."""

    @abstractmethod
    def is_applicable(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> bool:
        """Check if this generator can be applied to the given scene."""
        pass

    @abstractmethod
    def generate(
        self, scene: SceneInfo, objects: List[ObjectInfo], frame: FrameInfo = None
    ) -> List[Tuple[TQ, TA]]:
        """Generate VQA samples from scene, objects, and optionally a specific frame."""
        pass

    @abstractmethod
    def get_metric_class(self) -> str:
        """Return the name of the metric class to use for evaluation."""
        pass

    @abstractmethod
    def get_question_type(self) -> Type[TQ]:
        """Return the question type class used by this generator."""
        pass

    @abstractmethod
    def get_answer_type(self) -> Type[TA]:
        """Return the answer type class used by this generator."""
        pass
