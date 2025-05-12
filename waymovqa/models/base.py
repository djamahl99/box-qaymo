from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np


class Model:
    """Base class for all models."""
    prompts = []

    def __init__(self):
        pass
    
    def evaluate(self):
        pass
    
