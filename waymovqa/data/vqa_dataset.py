from typing import List, Dict, Optional, Tuple, Union, Type

from waymovqa.answers.base import BaseAnswer
from waymovqa.answers import answer_from_dict, answer_from_json
from waymovqa.questions import question_from_dict, question_from_json
from waymovqa.questions.base import BaseQuestion

import pprint

from pydantic import BaseModel

import json


class VQASample(BaseModel):
    question: BaseQuestion
    answer: Optional[BaseAnswer] = None

    def model_dump_json(self):
        return {
            "question": self.question.model_dump_json(),
            "answer": self.answer.model_dump_json() if self.answer else None,
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            question=question_from_json(data["question"]),
            answer=(
                answer_from_json(data["answer"]) if data["answer"] is not None else None
            ),
        )


class VQADataset:
    def __init__(self, tag: str = "ground_truth") -> None:
        self.samples: List[VQASample] = []
        self.tag = tag

    def add_sample(self, question: BaseQuestion, answer: Optional[BaseAnswer] = None):
        sample = VQASample(question=question, answer=answer)
        self.samples.append(sample)

    def save_dataset(self, path: str):
        serializable_samples = [sample.model_dump_json() for sample in self.samples]
        with open(path, "w") as f:
            json.dump({"tag": self.tag, "samples": serializable_samples}, f, indent=2)

    @classmethod
    def load_dataset(cls, path: str) -> "VQADataset":
        with open(path, "r") as f:
            data = json.load(f)

        dataset = cls(tag=data.get("tag", "unknown"))
        for entry in data["samples"]:
            sample = VQASample.from_json(entry)
            dataset.samples.append(sample)

        return dataset
