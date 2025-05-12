from .base import BaseQuestion

class SingleImageQuestion(BaseQuestion):
    image_path: str
    question: str
    