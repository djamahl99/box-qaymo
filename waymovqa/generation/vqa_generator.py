from typing import Dict, List, Any, Optional, Union
import importlib
import pkgutil
import inspect
from ..prompts import get_all_prompt_generators, get_prompt_generator, BasePromptGenerator

class VQAGenerator:
    """Main class for generating VQA samples."""
    
    def __init__(self, prompt_generators: Optional[Union[List[str], List[BasePromptGenerator]]] = None):
        """
        Initialize VQA generator.
        
        Args:
            prompt_generators: Optional list of prompt generator names or instances.
                               If None, all registered generators will be used.
        """
        if prompt_generators is None:
            # Use all registered prompt generators
            self.prompt_generators = [cls() for cls in get_all_prompt_generators().values()]
        else:
            self.prompt_generators = []
            for gen in prompt_generators:
                if isinstance(gen, str):
                    # Get the generator by name
                    self.prompt_generators.append(get_prompt_generator(gen)())
                elif isinstance(gen, BasePromptGenerator):
                    # Use the provided instance
                    self.prompt_generators.append(gen)
                else:
                    raise TypeError(f"Expected str or BasePromptGenerator, got {type(gen)}")
    
    def generate_dataset(self, scenes: List[Dict[str, Any]], questions_per_scene: int = 5) -> List[Dict[str, Any]]:
        """
        Generate VQA dataset from a list of scenes.
        
        Args:
            scenes: List of scene data dictionaries
            questions_per_scene: Number of questions to generate per scene
            
        Returns:
            List of VQA samples
        """
        dataset = []
        
        for scene in scenes:
            # Find applicable prompt generators for this scene
            applicable_generators = [
                gen for gen in self.prompt_generators 
                if gen.is_applicable(scene)
            ]
            
            scene_questions = []
            for generator in applicable_generators:
                # Calculate how many questions to generate from this prompt
                n = max(1, questions_per_scene // len(applicable_generators))
                scene_questions.extend(generator.generate_questions(scene, n))
            
            # Add scene information to each Q&A pair
            for qa in scene_questions:
                sample = {
                    "scene_id": scene.get("id", "unknown"),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "metadata": {
                        **qa.get("metadata", {}),
                        "prompt_type": type(generator).__name__
                    }
                }
                dataset.append(sample)
        
        return dataset