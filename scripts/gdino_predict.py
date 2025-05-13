import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from transformers import pipeline
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    AutoProcessor,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
)

from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.questions.single_image import SingleImageQuestion
from waymovqa.answers.object_2d import Object2DAnswer
from waymovqa.answers.multi_object_2d import MultiObject2DAnswer
from waymovqa.metrics.coco import COCOEvaluator
from waymovqa.waymo_loader import WaymoDatasetLoader

def convert_to_original_coords(boxes, original_shape=(1920, 1080), resized_shape=(1008, 1008)):
    """
    Convert bounding box coordinates from resized/padded image to original image size.

    Parameters:
    - boxes: Nx4 NumPy array of bounding boxes in the resized image [x_min, y_min, x_max, y_max].
    - original_shape: Tuple of (width, height) for the original image.
    - resized_shape: Tuple of (width, height) for the resized image.

    Returns:
    - original_boxes: Nx4 NumPy array of bounding boxes in the original image size.
    """
    original_shape = np.array(original_shape)
    resized_shape = np.array(resized_shape)

    ldim = np.argmax(original_shape)

    rev_scale_f = resized_shape[ldim] / original_shape[ldim]

    pad = resized_shape - original_shape * rev_scale_f  

    scale_w = (original_shape[0]) / (resized_shape[0] - pad[0])
    scale_h = (original_shape[1]) / (resized_shape[1] - pad[1])

    # Adjust coordinates
    boxes[:, [0, 2]] = (boxes[:, [0, 2]]) * scale_w
    boxes[:, [1, 3]] = (boxes[:, [1, 3]]) * scale_h


    # Clip coordinates to stay within original image boundaries
    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_shape[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_shape[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_shape[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_shape[1])

    return boxes

def run_on_dataset(
    gt_dataset: VQADataset,
    owlvit_model: OWLv2Detector,
    dataset_path: Path,
    save_path: Path,
    batch_size=2,
    conf_thresh=0.1,
    nms_thresh=0.5
) -> VQADataset:
    """
    Processes a single frame from the Waymo dataset using efficient batch processing.

    Args:
        dataset: VQADataset to run the model on
        sam_predictor: SAM predictor instance
        owlvit_model: OWLv2Detector model instance
        batch_size: Maximum number of images to process in a single batch
        conf_thresh: confidence threshold
        nms_thresh: NMS threshold
    """
    batch_images, image_metas, questions = convert_dataset(gt_dataset, dataset_path)

    pred_dataset = VQADataset(tag=f'owlpreds_{save_path.stem}')

    # Process images in smaller batches
    for batch_start in range(0, len(batch_images), batch_size):
        batch_end = min(batch_start + batch_size, len(batch_images))
        current_batch = batch_images[batch_start:batch_end]
        current_metas = image_metas[batch_start:batch_end]
        current_questions = questions[batch_start:batch_end]

        # Load the image_cv for the batch
        current_batch_loaded = []
        for image_path in current_batch:
            img_bgr = cv2.imread(image_path)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            current_batch_loaded.append(img)

        # Process the current batch
        owlvit_data = owlvit_model.preprocess_batch(current_batch_loaded)

        # Run the model on the batch
        with torch.no_grad():  # Disable gradient calculation for inference
            vit_outputs = owlvit_model.forward_batch(
                owlvit_data["pixel_values"].to("cuda:1"),
                target_size=[owlvit_model.image_size, owlvit_model.image_size],
            )

        # Process the results for each image in the current batch
        for i, (meta, question) in enumerate(zip(current_metas, current_questions)):
            # Extract this image's outputs
            image_outputs = {
                "pred_boxes": vit_outputs["pred_boxes"][i],
                "objectness_logits": vit_outputs["objectness_logits"][i],
                "image_class_embeds": vit_outputs["image_class_embeds"][i],
                "image_feats": vit_outputs["image_feats"][i],
            }

            # List of the single question/prompt given
            prompts = [question.question]

            # Convert logits to probabilities
            objectness = image_outputs["objectness_logits"].sigmoid()

            # Apply NMS
            nms_indices = nms(image_outputs["pred_boxes"], objectness, nms_thresh)

            print('image_outputs["pred_boxes"]', image_outputs["pred_boxes"].shape)
            # Extract selected boxes and scores
            selected_boxes = image_outputs["pred_boxes"][nms_indices].cpu().numpy()
            selected_objectness = objectness[nms_indices].cpu().numpy()
            print('selected_boxes', selected_boxes.shape)
            print('selected_objectness', selected_objectness.shape)

            num_boxes = selected_objectness.shape[0]

            # get the text features for this prompt
            prompt_text_features = owlvit_model.get_text_features(prompts)

            print('prompt_text_features', prompt_text_features.shape, prompt_text_features.device)
            image_class_embeds = image_outputs['image_class_embeds'][nms_indices].cpu()
            print('image_class_embeds', image_class_embeds.shape, image_class_embeds.device)
            
            prompt_out = torch.einsum("ij,kj->ik", image_class_embeds.detach().cpu(), prompt_text_features.detach().cpu())
            prompt_out = prompt_out.sigmoid().numpy()

            print('prompt_out', prompt_out.shape)
            print('selected_objectness', selected_objectness.shape)
            print('prompt_out', prompt_out.min(), prompt_out.max())
            print('prompt_out', selected_objectness.min(), selected_objectness.max())

            # TODO: update scoring?
            # final_scores = (prompt_out + selected_objectness.reshape(prompt_out.shape)) / 2.0
            final_scores = (prompt_out * selected_objectness.reshape(prompt_out.shape))

            print('final_scores', final_scores.shape)
            print('final_scores', final_scores.min(), final_scores.max())


            final_scores = final_scores.reshape(num_boxes, 1)
            final_boxes = selected_boxes.reshape(num_boxes, 4)

            final_thresh = max(conf_thresh, final_scores.min()) # at least one detection (min)
            # indices = torch.arange(num_boxes)[final_scores >= final_thresh]
            indices = np.nonzero(final_scores >= final_thresh)[0]

            final_scores = final_scores[indices]
            final_boxes = final_boxes[indices]

            print('final_scores', final_scores.shape)
            print('final_boxes', final_boxes.shape)

            # Convert to original image coordinates
            final_boxes = convert_to_original_coords(
                final_boxes.reshape(-1, 4),
                meta["shape"],
                [owlvit_model.image_size, owlvit_model.image_size],
            )

            if final_boxes.shape[0] == 0:
                answer = Object2DAnswer(box=[0.0, 0.0, 0.0, 0.0], score=0.0)
            elif final_boxes.shape[0] == 1:
                answer = Object2DAnswer(box=final_boxes[0].tolist(), score=float(final_scores[0]))
            else:
                answer = MultiObject2DAnswer(boxes=final_boxes.tolist(), scores=final_scores.reshape(-1).tolist())

            # Add predictions
            pred_dataset.add_sample(
                question,
                answer
            )

        # Clear CUDA cache between batches to free up memory
        if "cuda" in str(next(owlvit_model.parameters()).device):
            torch.cuda.empty_cache()

    # Convert results to numpy arrays
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
        description="Generate VQA dataset from processed data"
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
        "--save_path", type=str, required=True, help="Path to save vqa predictions dataset"
    )
    parser.add_argument(
        "--pretrained_name", type=str, required=False, default="google/owlv2-large-patch14-ensemble", help="OWLViT Pretrained model name"
    )
    parser.add_argument(
        "--nms_thresh", type=float, default=0.5, required=False, help="NMS Threshold"
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=0.1, required=False, help="Confidence Threshold"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, required=False, help="Batch size"
    )
    parser.add_argument(
        "--device", type=str, required=False, default="cuda:0", help="Torch device for model"
    )

    args = parser.parse_args()

    gt_dataset = VQADataset.load_dataset(args.vqa_path)

    save_path = Path(args.save_path)

    if not save_path.exists():
        owlvit_model = OWLv2Detector(args.pretrained_name, ['placeholder']).to(args.device)

        pred_dataset = run_on_dataset(gt_dataset, owlvit_model, Path(args.dataset_path), save_path, args.batch_size, args.conf_thresh, args.nms_thresh)
    else:
        pred_dataset = VQADataset.load_dataset(str(save_path))

    # evaluate?
    coco_eval = COCOEvaluator()

    predictions = [x[1] for x in pred_dataset.samples]
    ground_truths = [x[1] for x in gt_dataset.samples]

    # get the questions from both pred / gt to check they are matching
    prompts0 = [x[0].question for x in pred_dataset.samples]
    prompts1 = [x[0].question for x in gt_dataset.samples]

    print('prompts0', len(prompts0))
    print('prompts1', len(prompts1))
    assert all([prompts0[i] == prompts1[i] for i in range(len(prompts0))]) and len(prompts0) == len(prompts1)

    pprint.pprint(coco_eval.evaluate(predictions, ground_truths, prompts0))

if __name__ == "__main__":
    main()
