# Import the necessary utilities first
from waymovqa.prompt_generators import register_prompt_generator

# Then import your modules
from .object_binary import ObjectBinaryPromptGenerator

# from .object_location_regression import SingleImageObjectLocationPromptGenerator
from .object_drawn_box_prompt import ObjectDrawnBoxPromptGenerator

from .ego_relative_trajectory import EgoRelativeObjectTrajectoryPromptGenerator
