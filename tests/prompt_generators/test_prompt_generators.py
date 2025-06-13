import unittest
import inspect
import os
import importlib
import sys
from pathlib import Path
from typing import List, Set


# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)  # Adjust if needed
print("project_root", project_root)
# exit()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from box_qaymo.prompt_generators.base import BasePromptGenerator
from box_qaymo.prompt_generators import get_all_prompt_generators
from box_qaymo.metrics.base import BaseMetric


class TestPromptGeneratorInterface(unittest.TestCase):
    """Test that all prompt generators implement the required methods."""

    def test_all_prompt_generators_have_required_methods(self):
        """
        Check that all prompt generators in box_qaymo/prompt_generators and sub-folders
        implement the required methods.
        """
        # Required methods to check
        required_methods = [
            "get_metric_class",
            "get_question_type",
            "get_answer_type",
        ]

        # Import all modules in prompt_generators directory and subdirectories
        generators = self._get_all_generator_classes()

        # Check each generator class
        for gen_class in generators:
            for method_name in required_methods:
                # Check if the method exists and is implemented (not abstract)
                method = getattr(gen_class, method_name, None)
                self.assertIsNotNone(
                    method, f"Method {method_name} not found in {gen_class.__name__}"
                )

                # Check if the method is not abstract
                if (
                    hasattr(method, "__isabstractmethod__")
                    and method.__isabstractmethod__
                ):
                    self.fail(
                        f"Method {method_name} in {gen_class.__name__} is abstract"
                    )

                # Check if method is inherited from BasePromptGenerator or implemented
                # in the class itself
                base_method = getattr(BasePromptGenerator, method_name, None)
                if base_method is not None:
                    # If method is same as in base class, it's using the default implementation
                    class_method = inspect.getattr_static(gen_class, method_name, None)
                    if class_method is inspect.getattr_static(
                        BasePromptGenerator, method_name, None
                    ):
                        # Just a warning, not a failure, as default implementations are OK
                        print(
                            f"Warning: {gen_class.__name__} uses default implementation of {method_name}"
                        )

    def _get_all_generator_classes(self) -> List[type]:
        """
        Get all prompt generator classes from the box_qaymo/prompt_generators directory
        and its subdirectories.
        """
        # Use the registry if available
        registered_generators = get_all_prompt_generators()
        if registered_generators:
            return list(registered_generators.values())

        # If registry is not available or empty, discover classes by scanning directories
        generators = []
        prompt_generators_dir = Path(project_root) / "box_qaymo" / "prompt_generators"

        # Walk through all subdirectories
        for root, dirs, files in os.walk(prompt_generators_dir):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    module_path = os.path.join(root, file)
                    # Convert file path to module import path
                    relative_path = os.path.relpath(module_path, project_root)
                    import_path = relative_path.replace(os.path.sep, ".").replace(
                        ".py", ""
                    )

                    try:
                        # Import the module
                        module = importlib.import_module(import_path)

                        # Find all prompt generator classes in the module
                        for name, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and issubclass(obj, BasePromptGenerator)
                                and obj != BasePromptGenerator
                            ):
                                generators.append(obj)
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Could not import {import_path}: {e}")

        return generators

    def test_generator_implementations_work(self):
        """
        Test that the method implementations actually work by calling them
        and ensuring they return appropriate values.
        """
        generators = self._get_all_generator_classes()

        print("generators", generators)

        for gen_class in generators:
            gen_instance = gen_class()

            # Test get_metric_class
            metric_class = gen_instance.get_metric_class()
            self.assertTrue(
                isinstance(metric_class, BaseMetric),
                f"{gen_class.__name__}.get_metric_class() should return a BaseMetric instance or subclass",
            )

            # Test get_question_type
            question_type = gen_instance.get_question_type()
            self.assertIsNotNone(
                question_type,
                f"{gen_class.__name__}.get_question_type() should not return None",
            )

            # Test get_answer_type
            answer_type = gen_instance.get_answer_type()
            self.assertIsNotNone(
                answer_type,
                f"{gen_class.__name__}.get_answer_type() should not return None",
            )


if __name__ == "__main__":
    unittest.main()
