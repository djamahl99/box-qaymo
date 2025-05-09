from typing import Dict, List, Any
from ..base import BasePromptGenerator, register_prompt_generator

@register_prompt_generator
class ClassCountPromptGenerator(BasePromptGenerator):
    """Generate questions about vehicles in the scene."""
    
    def is_applicable(self, scene_data: Dict[str, Any]) -> bool:
        """Check if there are vehicles in the scene."""
        # return any(obj["category"] == "vehicle" for obj in scene_data.get("objects", []))
        return any(obj for obj in scene_data.get("objects", []))
    
    def generate_questions(self, scene_data: Dict[str, Any], n_questions: int = 1) -> List[Dict[str, Any]]:
        """Generate questions about vehicles in the scene."""
        # Count the number of each object + each colour etc?
        
        for obj in scene_data.get("objects", []):
            pass