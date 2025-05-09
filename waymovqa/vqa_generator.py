import json
from typing import Dict, List, Any, Optional, Union
from .prompts import (
    get_all_prompt_generators,
    get_prompt_generator,
    BasePromptGenerator,
)
from pathlib import Path

from waymo_data import ObjectInfo


class VQAGenerator:
    """Main class for generating VQA samples."""

    def __init__(
        self,
        dataset_path: Path,
        prompt_generators: Optional[Union[List[str], List[BasePromptGenerator]]] = None,
    ):
        """
        Initialize VQA generator.

        Args:
            prompt_generators: Optional list of prompt generator names or instances.
                               If None, all registered generators will be used.
        """
        self.dataset_path = dataset_path

        if prompt_generators is None:
            # Use all registered prompt generators
            self.prompt_generators = [
                cls() for cls in get_all_prompt_generators().values()
            ]
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
                    raise TypeError(
                        f"Expected str or BasePromptGenerator, got {type(gen)}"
                    )
                    
    def find_scenes(self) -> List[str]:
        """
            Finds scenes to sample from.
        """
        scenes_path = self.dataset_path / 'scene_infos'
        
        scenes = [x.stem for x in scenes_path.rglob('*.json')]
                    
        return scenes
                    
    def find_scene_objects(self, scene_id: str) -> List[ObjectInfo]:
        """
            Given a scene_id it will find all the ObjectInfo.
            
            Arguments:
                scene_id: str - the scene_id to find the objects for.
        """
        object_lists_path = self.dataset_path / 'object_lists'
        all_object_table_path = object_lists_path / f'{scene_id}_all_object_table.json'
        
        assert all_object_table_path.exists(), 'All object table does not exist!'
        
        with open(all_object_table_path, 'r') as f:
            all_object_table = json.load(f)


        objects = list(map(ObjectInfo.from_dict, all_object_table))
        
        return objects

    def generate_dataset(self, questions_per_scene: int = 5) -> List[Dict[str, Any]]:
        """
        Generate VQA dataset.

        Args:
            questions_per_scene: Number of questions to generate per scene

        Returns:
            List of VQA samples
        """
        pass