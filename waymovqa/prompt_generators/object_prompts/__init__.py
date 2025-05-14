# Import the necessary utilities first
from waymovqa.prompt_generators import register_prompt_generator

# Then import your modules
from .grounding_2d import Grounding2DPromptGenerator
from .object_color import *
from .object_location import *
from .object_relation import *