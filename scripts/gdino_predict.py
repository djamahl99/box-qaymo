import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Tuple, Optional, List
from argparse import ArgumentParser
import torch
from torch import nn
from torchvision.ops import box_iou, nms
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import pprint

from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.utils import clean_state_dict

from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.questions.single_image_single_object import (
    SingleImageSingleObjectQuestion,
)
from waymovqa.answers.object_2d import Object2DAnswer
from waymovqa.answers.multi_object_2d import MultiObject2DAnswer
from waymovqa.metrics.coco import COCOEvaluator
from waymovqa.waymo_loader import WaymoDatasetLoader


class GroundingDINODetector:
    def __init__(self, model_config_path, model_checkpoint_path, device="cuda:0"):
        self.device = device
        self.image_size = 800  # Default GroundingDINO size

        # Load model
        self.model = GroundingDINO(
            config_file=model_config_path,
            checkpoint_path=model_checkpoint_path,
        )
        self.model.to(device)
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def preprocess_batch(self, images):
        processed_images = []
        for image in images:
            # Convert numpy to PIL
            image_pil = Image.fromarray(image)
            # Resize maintaining aspect ratio
            processed_images.append(image_pil)

        return {"images": processed_images}

    def forward_batch(self, images, target_size=None):
        results = []
        for image in images:
            # Process each image individually
            boxes, logits = self.model.predict_with_caption(
                image=image,
                caption="",  # Will be set during inference
                box_threshold=0.05,  # Low threshold, will be filtered later
                text_threshold=0.05,
            )
            results.append({"pred_boxes": boxes, "scores": logits})

        return results

    def get_text_features(self, text_prompts):
        # Get text embeddings for the prompts
        text_features = []
        for prompt in text_prompts:
            text_feature = self.model.get_text_embeddings(prompt)
            text_features.append(text_feature)

        return torch.stack(text_features).to(self.device)


def run_on_dataset(
    gt_dataset: VQADataset,
    dino_model: GroundingDINODetector,
    dataset_path: Path,
    save_path: Path,
    batch_size=2,
    conf_thresh=0.1,
    nms_thresh=0.5,
) -> VQADataset:
    """
    Processes a dataset using Grounding DINO for object detection.
    """
    batch_images, image_metas, questions = convert_dataset(gt_dataset, dataset_path)
    pred_dataset = VQADataset(tag=f"dinopreds_{save_path.stem}")

    # Process images in smaller batches
    for batch_start in range(0, len(batch_images), batch_size):
        batch_end = min(batch_start + batch_size, len(batch_images))
        current_batch = batch_images[batch_start:batch_end]
        current_metas = image_metas[batch_start:batch_end]
        current_questions = questions[batch_start:batch_end]

        # Load the images for the batch
        current_batch_loaded = []
        for image_path in current_batch:
            img_bgr = cv2.imread(str(image_path))
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            current_batch_loaded.append(img)

        # Process the current batch
        processed_data = dino_model.preprocess_batch(current_batch_loaded)

        # Process each image in the batch
        for i, (img, meta, question) in enumerate(
            zip(processed_data["images"], current_metas, current_questions)
        ):
            # Get prompt from question
            prompt = question.question

            # Run GroundingDINO on the image with the prompt
            boxes, scores = dino_model.model.predict_with_caption(
                image=img,
                caption=prompt,
                box_threshold=conf_thresh,
                text_threshold=conf_thresh,
            )

            # Convert boxes to numpy array
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()

            # Apply NMS if needed
            if len(boxes_np) > 1:
                # Convert to torch tensor for NMS
                boxes_torch = torch.from_numpy(boxes_np).to(dino_model.device)
                scores_torch = torch.from_numpy(scores_np).to(dino_model.device)

                # Apply NMS
                nms_indices = nms(boxes_torch, scores_torch, nms_thresh)

                # Extract selected boxes and scores
                selected_boxes = boxes_np[nms_indices.cpu().numpy()]
                selected_scores = scores_np[nms_indices.cpu().numpy()]
            else:
                selected_boxes = boxes_np
                selected_scores = scores_np

            # Convert to original image coordinates (input is already normalized to [0,1])
            final_boxes = selected_boxes.copy()
            original_shape = meta["shape"]  # [width, height]

            # Denormalize coordinates to absolute pixel values
            final_boxes[:, 0] *= original_shape[0]  # x1 * width
            final_boxes[:, 1] *= original_shape[1]  # y1 * height
            final_boxes[:, 2] *= original_shape[0]  # x2 * width
            final_boxes[:, 3] *= original_shape[1]  # y2 * height

            # Create answer object
            if len(final_boxes) == 0:
                answer = Object2DAnswer(box=[0.0, 0.0, 0.0, 0.0], score=0.0)
            elif len(final_boxes) == 1:
                answer = Object2DAnswer(
                    box=final_boxes[0].tolist(), score=float(selected_scores[0])
                )
            else:
                answer = MultiObject2DAnswer(
                    boxes=final_boxes.tolist(),
                    scores=selected_scores.reshape(-1).tolist(),
                )

            # Add predictions
            pred_dataset.add_sample(question, answer)

        # Clear CUDA cache between batches
        if "cuda" in dino_model.device:
            torch.cuda.empty_cache()

    # Save dataset
    pred_dataset.save_dataset(str(save_path))
    return pred_dataset


def convert_dataset(dataset: VQADataset, dataset_path: Path):
    batch_images = []
    image_metas = []
    questions = []

    for question, answer in dataset.samples:
        assert isinstance(question, SingleImageQuestion) and isinstance(
            answer, Object2DAnswer
        )

        # probably should make this more succinct
        image_path = dataset_path / question.image_path

        with Image.open(image_path) as img:
            width, height = img.size

        batch_images.append(image_path)
        image_metas.append(
            {
                "cv_image": None,
                "shape": [width, height],  # width, height
                "name": Path(image_path.name),
            }
        )
        questions.append(question)

    return batch_images, image_metas, questions


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VQA dataset from processed data using Grounding DINO"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to processed waymo dataset",
    )
    parser.add_argument(
        "--vqa_path", type=str, required=True, help="Path to vqa gt dataset"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save vqa predictions dataset",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to GroundingDINO config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to GroundingDINO checkpoint",
    )
    parser.add_argument(
        "--nms_thresh", type=float, default=0.5, required=False, help="NMS Threshold"
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.1,
        required=False,
        help="Confidence Threshold",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, required=False, help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="Torch device for model",
    )

    args = parser.parse_args()

    gt_dataset = VQADataset.load_dataset(args.vqa_path)
    save_path = Path(args.save_path)

    if not save_path.exists():
        dino_model = GroundingDINODetector(
            args.config_path, args.checkpoint_path, device=args.device
        )

        pred_dataset = run_on_dataset(
            gt_dataset,
            dino_model,
            Path(args.dataset_path),
            save_path,
            args.batch_size,
            args.conf_thresh,
            args.nms_thresh,
        )
    else:
        pred_dataset = VQADataset.load_dataset(str(save_path))

    # evaluate?
    coco_eval = COCOEvaluator()

    predictions = [x[1] for x in pred_dataset.samples]
    ground_truths = [x[1] for x in gt_dataset.samples]

    # get the questions from both pred / gt to check they are matching
    prompts0 = [x[0].question for x in pred_dataset.samples]
    prompts1 = [x[0].question for x in gt_dataset.samples]

    print("prompts0", len(prompts0))
    print("prompts1", len(prompts1))
    assert all([prompts0[i] == prompts1[i] for i in range(len(prompts0))]) and len(
        prompts0
    ) == len(prompts1)

    pprint.pprint(coco_eval.evaluate(predictions, ground_truths, prompts0))


if __name__ == "__main__":
    main()
