# Import the necessary utilities first
from waymovqa.prompt_generators import register_prompt_generator

# Then import your modules
from .grounding_2d import Grounding2DPromptGenerator
from .object_color import *
from .object_location_regression import *
from .object_relation import *
from .object_drawn_box_prompt import ObjectDrawnBoxPromptGenerator

# from .object_trajectory import ObjectTrajectoryPromptGenerator
from .object_trajectory_relative import ObjectTrajectoryPromptGenerator
