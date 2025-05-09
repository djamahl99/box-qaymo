from .base import PROMPT_REGISTRY, BasePromptGenerator, register_prompt_generator

# Import all prompt generators to ensure they're registered
from .object_prompts import *

# Function to get all registered prompt generators
def get_all_prompt_generators():
    """Return a dictionary of all registered prompt generators."""
    return PROMPT_REGISTRY

# Function to get a specific prompt generator by name
def get_prompt_generator(name):
    """
    Get a prompt generator by name.
    
    Args:
        name: Name of the prompt generator class
        
    Returns:
        The prompt generator class
        
    Raises:
        KeyError: If the prompt generator is not found
    """
    if name not in PROMPT_REGISTRY:
        raise KeyError(f"Prompt generator '{name}' not found in registry")
    return PROMPT_REGISTRY[name]