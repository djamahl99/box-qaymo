from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional

from waymovqa.data import FrameInfo, ObjectInfo, SceneInfo
from waymovqa.eval.answer import BaseAnswer

# Update BasePromptGenerator with typing information
T = TypeVar('T', bound=BaseAnswer)

# Registry to store all prompt generators
PROMPT_REGISTRY = {}


def register_prompt_generator(cls):
    """Decorator to register a prompt generator class in the registry."""
    PROMPT_REGISTRY[cls.__name__] = cls
    return cls

def get_prompt_generator(name: str):
    """Get prompt generator by name."""
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt generator '{name}' not found")
    return PROMPT_REGISTRY[name]

def get_all_prompt_generators():
    """Get all registered prompt generators."""
    return PROMPT_REGISTRY


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
    ) -> List[Dict[str, Any]]:
        """Generate VQA samples from scene, objects, and optionally a specific frame."""
        pass
    
    @abstractmethod
    def get_metric_class(self) -> str:
        """Return the name of the metric class to use for evaluation."""
        pass
    
    @abstractmethod
    def get_answer_type(self) -> Type[T]:
        """Return the answer type class used by this generator."""
        pass
    
    @abstractmethod
    def parse_answer(self, answer_text: str) -> T:
        """Parse a textual answer into the structured format."""
        pass
