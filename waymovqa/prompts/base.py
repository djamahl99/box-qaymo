from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type

# Registry to store all prompt generators
PROMPT_REGISTRY = {}

def register_prompt_generator(cls):
    """Decorator to register a prompt generator class in the registry."""
    PROMPT_REGISTRY[cls.__name__] = cls
    return cls

class BasePromptGenerator(ABC):
    """
    Abstract base class for all prompt generators.
    Defines the interface that all prompt generators must implement.
    """
    
    @abstractmethod
    def is_applicable(self, scene_data: Dict[str, Any]) -> bool:
        """
        Check if this prompt generator can be applied to the given scene.
        
        Args:
            scene_data: Dictionary containing scene metadata and object information
            
        Returns:
            bool: True if this prompt generator can be applied, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_questions(self, scene_data: Dict[str, Any], n_questions: int = 1) -> List[Dict[str, Any]]:
        """
        Generate VQA questions for the given scene.
        
        Args:
            scene_data: Dictionary containing scene metadata and object information
            n_questions: Number of questions to generate
            
        Returns:
            List of dictionaries, each containing:
                - question: str, the generated question
                - answer: str, the answer to the question
                - metadata: dict, additional metadata about the Q&A pair
        """
        pass