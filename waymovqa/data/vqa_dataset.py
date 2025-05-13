from typing import List, Dict, Tuple, Union, Type

from waymovqa.answers.base import BaseAnswer
from waymovqa.answers import answer_from_dict, answer_from_json
from waymovqa.questions import question_from_dict, question_from_json
from waymovqa.questions.base import BaseQuestion

import json

class VQADataset:
    def __init__(self, tag: str = "ground_truth") -> None:
        self.samples: List[Tuple[BaseQuestion, BaseAnswer]] = []
        self.tag = tag

    def add_sample(self, question: BaseQuestion, answer: BaseAnswer):
        self.samples.append((question, answer))

    def save_dataset(self, path: str):
        serializable_samples = []
        for question, answer in self.samples:
            serializable_samples.append({
                "question": question.model_dump_json(),
                "answer": answer.model_dump_json(),
            })

        with open(path, "w") as f:
            json.dump({
                "tag": self.tag,
                "samples": serializable_samples
            }, f, indent=2)

    @classmethod
    def load_dataset(
        cls,
        path: str
    ) -> "VQADataset":
        with open(path, "r") as f:
            data = json.load(f)

        dataset = cls(tag=data.get("tag", "unknown"))
        for entry in data["samples"]:
            question = question_from_json(entry["question"])

            print('entry["answer"]', entry["answer"])

            answer = answer_from_json(entry["answer"])

            dataset.add_sample(question, answer)

        return dataset
