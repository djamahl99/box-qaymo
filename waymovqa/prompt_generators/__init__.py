# First define the registry and registration function
PROMPT_REGISTRY = {}


def register_prompt_generator(cls):
    """Decorator to register a prompt generator class in the registry."""
    PROMPT_REGISTRY[cls.__name__] = cls
    return cls


# Import the base class
from .base import BasePromptGenerator

# Then import the prompt generators
from .object_prompts import *
from .scene_prompts import *


def get_prompt_generator(name: str):
    """Get prompt generator by name."""
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt generator '{name}' not found")
    return PROMPT_REGISTRY[name]


def get_all_prompt_generators():
    """Get all registered prompt generators."""
    return PROMPT_REGISTRY
