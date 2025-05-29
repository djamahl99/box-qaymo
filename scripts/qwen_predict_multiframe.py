import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Tuple, Optional, List
import torch
from pathlib import Path
from PIL import Image
import pprint
import re
import json
import cv2

from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.questions import *
from waymovqa.answers.base import BaseAnswer
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.metrics.multiple_choice import MultipleChoiceMetric
from waymovqa.answers.raw_text import RawTextAnswer
from waymovqa.questions.single_image import SingleImageQuestion
from waymovqa.questions.single_image_multi_prompt import SingleImageMultiplePromptQuestion
from waymovqa.questions.single_image_multi_choice import SingleBase64ImageMultipleChoiceQuestion
from waymovqa.questions.single_image_multi_choice import SingleImageMultipleChoiceQuestion
from waymovqa.questions.multi_image import MultipleImageQuestion
from waymovqa.questions.multi_image_multi_choice import MultipleImageMultipleChoiceQuestion
from waymovqa.questions.multi_frame_multi_choice import MultipleFrameMultipleChoiceQuestion

from transformers import AutoTokenizer
import torch
from PIL import Image

import argparse
import torch

from PIL import Image
import random

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from pathlib import Path
from typing import List, Union
import gc


def load_image(image_path, bbox=None, draw_bbox=True):
    """Load and optionally draw bbox on image"""
    if not draw_bbox or bbox is None:
        return Image.open(image_path)

    img_vis = cv2.imread(str(image_path))
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 6)

    # Convert back to PIL Image to match original function
    pil_img = Image.fromarray(img_vis)
    return pil_img


def format_options(choices: List[str]):
    return ", ".join(choices[:-1]) + " or " + choices[-1]


def get_prompt(question: BaseQuestion, gt_answer: BaseAnswer) -> str:
    """Get the prompt"""
    extra_text = ""
    choices = []
    if isinstance(question, SingleImageMultipleChoiceQuestion):
        choices = question.choices
    elif isinstance(question, SingleImageQuestion) and isinstance(
        gt_answer, MultipleChoiceAnswer
    ):
        choices = gt_answer.choices
    elif isinstance(question, MultipleImageMultipleChoiceQuestion):
        best_camera_name = "FRONT"
        if isinstance(question.data, dict) and "best_camera_name" in question.data:
            best_camera_name = question.data["best_camera_name"]

        choices = gt_answer.choices
        extra_text = f"Focus on the {best_camera_name.lower().replace('_', '')} camera."
    elif isinstance(question, MultipleImageQuestion) and isinstance(
        gt_answer, MultipleChoiceAnswer
    ):
        best_camera_name = "FRONT"
        if isinstance(question.data, dict) and "best_camera_name" in question.data:
            best_camera_name = question.data["best_camera_name"]

        choices = gt_answer.choices
        extra_text = f"Focus on the {best_camera_name.lower().replace('_', '')} camera."
    elif isinstance(question, MultipleFrameMultipleChoiceQuestion) and isinstance(
        gt_answer, MultipleChoiceAnswer
    ):
        choices = gt_answer.choices
    else:
        raise TypeError(f"question->{type(question)} answer->{type(gt_answer)}")

    prompt = f'{question.question}\nRespond with only the full name of your selection (e.g., "{random.choice(choices)}").'
    prompt += f"Choose the best option out of {format_options(choices)}."
    prompt += extra_text

    return prompt


def run_on_dataset(
    gt_dataset,  # VQADataset type
    args,
    dataset_path: Path,
    save_path: Path,
    batch_size=1,
    draw_bbox=True,
):
    """
    Process a dataset with Qwen-VL model.
    
    Args:
        gt_dataset: VQADataset to run the model on
        args: Arguments for Qwen-VL
        dataset_path: path to extracted data
        save_path: path to save results
        batch_size: Number of samples to process (Qwen-VL processes one at a time)
    
    Returns:
        VQADataset with predictions
    """
    # Load model and tokenizer once
    print(f"Loading Qwen-VL model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if hasattr(args, 'fp16') and args.fp16 else torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    pred_dataset = VQADataset(tag=f"qwen_{save_path.stem}")
    
    print(f"Processing {len(gt_dataset.samples)} samples...")
    
    for i, sample in enumerate(gt_dataset.samples):
        question = sample.question
        gt_answer = sample.answer
        
        # Handle different question types
        if isinstance(question, MultipleFrameMultipleChoiceQuestion):
            # Process multiframe questions
            prompt = get_prompt(question, gt_answer)
            
            image_paths = [
                str(dataset_path / question.previous_image_path), 
                str(dataset_path / question.current_image_path)
            ]
            bboxes = [question.previous_bbox, question.current_bbox]
            
            pred_answer_text = process_multiframe_sample(
                prompt, image_paths, bboxes, model, tokenizer, args, draw_bbox
            )
            
        elif isinstance(question, SingleImageMultipleChoiceQuestion):
            image_path = str(dataset_path / question.image_path)
            prompt = get_prompt(question, gt_answer)
            
            bbox = None
            if isinstance(question.data, dict) and "bbox" in question.data:
                bbox = question.data['bbox']
            
            pred_answer_text = process_single_sample(
                prompt, image_path, bbox, model, tokenizer, args, draw_bbox
            )

        elif isinstance(question, SingleImageQuestion) and isinstance(
            gt_answer, MultipleChoiceAnswer
        ):
            prompt = get_prompt(question, gt_answer)
            # Handle single image questions without explicit image path
            pred_answer_text = process_single_sample(
                prompt, None, None, model, tokenizer, args, draw_bbox
            )

        elif isinstance(question, MultipleImageMultipleChoiceQuestion):
            best_camera_name = "FRONT"
            if isinstance(question.data, dict) and "best_camera_name" in question.data:
                best_camera_name = question.data['best_camera_name']
                
            cam_idx = next((i for i, cam_name in enumerate(question.camera_names) if cam_name == best_camera_name), 0)
            image_path = str(dataset_path / question.image_paths[cam_idx])
            prompt = get_prompt(question, gt_answer)
            
            bbox = None
            if isinstance(question.data, dict) and "bbox" in question.data:
                bbox = question.data['bbox']
            
            pred_answer_text = process_single_sample(
                prompt, image_path, bbox, model, tokenizer, args, draw_bbox
            )

        elif isinstance(question, MultipleImageQuestion) and isinstance(gt_answer, MultipleChoiceAnswer):
            best_camera_name = "FRONT"
            if isinstance(question.data, dict) and "best_camera_name" in question.data:
                best_camera_name = question.data['best_camera_name']
                
            cam_idx = next((i for i, cam_name in enumerate(question.camera_names) if cam_name == best_camera_name), 0)
            image_path = str(dataset_path / question.image_paths[cam_idx])
            prompt = get_prompt(question, gt_answer)
            
            bbox = None
            if isinstance(question.data, dict) and "bbox" in question.data:
                bbox = question.data['bbox']
            
            pred_answer_text = process_single_sample(
                prompt, image_path, bbox, model, tokenizer, args, draw_bbox
            )
            
        else:
            print(f"Unsupported question type: {type(question)}")
            raise TypeError(
                f"Question type {question.__class__.__name__} not valid for this model"
            )
        
        # Create prediction answer
        pred_answer = RawTextAnswer(text=pred_answer_text)
        pred_dataset.add_sample(question, pred_answer)
        
        # Progress tracking
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(gt_dataset.samples)} samples")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save results
    pred_dataset.save_dataset(str(save_path))
    print(f"Results saved to {save_path}")
    
    return pred_dataset


def process_multiframe_sample(prompt, image_paths, bboxes, model, tokenizer, args, draw_bbox=True):
    """
    Process a multiframe sample with Qwen-VL.
    
    Args:
        prompt: Text prompt
        image_paths: List of image paths
        bboxes: List of bounding boxes (can contain None values)
        model: Qwen-VL model
        tokenizer: Qwen-VL tokenizer
        args: Configuration arguments
        draw_bbox: Whether to draw bounding boxes
    
    Returns:
        Generated response text
    """
    # Process images with bboxes if needed
    processed_images = []
    temp_image_paths = []
    
    dataset_path = Path("/scratch/user/uqdetche/waymo_vqa_dataset/camera_images/")
    
    for i, (image_path, bbox) in enumerate(zip(image_paths, bboxes)):
        image_path = str(dataset_path / Path(image_path).name)
        
        assert Path(image_path).exists():
            
        if bbox is not None and draw_bbox:
            # Create image with bbox drawn
            img_with_bbox = load_image(image_path, bbox, draw_bbox=True)
            temp_path = f'./temp_bbox_{i}.jpg'
            img_with_bbox.save(temp_path)
            temp_image_paths.append(temp_path)
            processed_images.append(temp_path)
        else:
            processed_images.append(image_path)
    
    if not processed_images:
        raise ValueError("No valid images found for multiframe question")
    
    try:
        # Prepare the query with multiple images
        query_list = []
        
        # Add images first
        for img_path in processed_images:
            query_list.append({'image': img_path})
        
        # Add the text prompt
        query_list.append({'text': prompt})
        
        query = tokenizer.from_list_format(query_list)
        
        # Generate response
        with torch.inference_mode():
            response, history = model.chat(
                tokenizer, 
                query, 
                history=None,
                temperature=getattr(args, 'temperature', 0.2),
                max_new_tokens=getattr(args, 'max_new_tokens', 512),
                do_sample=getattr(args, 'do_sample', True),
            )
        
        return response
        
    finally:
        # Clean up temporary files
        for temp_path in temp_image_paths:
            if Path(temp_path).exists():
                os.remove(temp_path)


def process_single_sample(prompt, image_path, bbox, model, tokenizer, args, draw_bbox=True):
    """
    Process a single sample with Qwen-VL.
    
    Args:
        prompt: Text prompt
        image_path: Path to image (can be None)
        bbox: Bounding box (can be None)
        model: Qwen-VL model
        tokenizer: Qwen-VL tokenizer
        args: Configuration arguments
        draw_bbox: Whether to draw bounding boxes
    
    Returns:
        Generated response text
    """
    temp_path = None
    
    try:
        # Handle image processing
        if image_path is not None and Path(image_path).exists():
            # Check if we need to draw bbox
            if bbox is not None and draw_bbox:
                img_with_bbox = load_image(image_path, bbox, draw_bbox=True)
                temp_path = './temp_single_bbox.jpg'
                img_with_bbox.save(temp_path)
                final_image_path = temp_path
            else:
                final_image_path = image_path
            
            # Prepare the query with image
            query = tokenizer.from_list_format([
                {'image': final_image_path},
                {'text': prompt},
            ])
        else:
            # Text-only query
            query = prompt
        
        # Generate response
        with torch.inference_mode():
            response, history = model.chat(
                tokenizer, 
                query, 
                history=None,
                temperature=getattr(args, 'temperature', 0.2),
                max_new_tokens=getattr(args, 'max_new_tokens', 512),
                do_sample=getattr(args, 'do_sample', True),
            )
        
        return response
        
    finally:
        # Clean up temporary file
        if temp_path and Path(temp_path).exists():
            os.remove(temp_path)


def create_args_for_qwen(model_path="Qwen/Qwen-VL-Chat", **kwargs):
    """
    Create an args object for Qwen-VL with default values.
    
    Args:
        model_path: Path to Qwen-VL model
        **kwargs: Additional arguments to override defaults
    
    Returns:
        Namespace object with configuration
    """
    from argparse import Namespace
    
    default_args = {
        'model_path': model_path,
        'temperature': 0.2,
        'max_new_tokens': 512,
        'do_sample': True,
        'fp16': True,  # Use half precision for memory efficiency
    }
    
    # Override with any provided kwargs
    default_args.update(kwargs)
    
    return Namespace(**default_args)


def parse_multiple_choice_response(response: str, choices: List[str]) -> str:
    """
    Parse the response to extract the chosen answer for multiple choice questions.
    """
    response = response.lower().strip()
    normalized_choices = [choice.lower() for choice in choices]

    # Check for exact phrase matches first (most reliable)
    for i, choice in enumerate(normalized_choices):
        pattern = f"(^|[^a-z0-9])(option|choice|answer)?\s*{i+1}([^a-z0-9]|$)"
        if re.search(pattern, response):
            return choices[i]

    # Check for exact choice text
    for i, choice in enumerate(normalized_choices):
        if re.search(f"(^|[^a-z0-9]){re.escape(choice)}([^a-z0-9]|$)", response):
            return choices[i]

    # Check for partial matches, sorted by length (longest first to avoid substring issues)
    sorted_choices = sorted(
        range(len(choices)), key=lambda i: len(normalized_choices[i]), reverse=True
    )
    for i in sorted_choices:
        if normalized_choices[i] in response:
            return choices[i]

    # Fallback to similarity for truly ambiguous cases
    return max(choices, key=lambda x: compute_text_similarity(x.lower(), response))


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple text similarity between two strings.
    """
    text1_tokens = set(text1.lower().split())
    text2_tokens = set(text2.lower().split())
    intersection = text1_tokens.intersection(text2_tokens)
    union = text1_tokens.union(text2_tokens)
    return len(intersection) / len(union) if union else 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Qwen-VL on VQA questions")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to processed waymo dataset",
    )
    parser.add_argument(
        "--vqa_path", type=str, required=False, help="Path to vqa gt dataset"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="Path to save vqa predictions dataset",
    )
    parser.add_argument(
        "--save_suffix",
        type=str,
        required=False,
        help="save_path suffix when running on multiple questions",
    )
    parser.add_argument(
        "--data_save_prefix",
        type=str,
        required=False,
        default="26_05_2025_export",
        help="save_path suffix when running on multiple questions",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="Qwen/Qwen-VL-Chat",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, required=False, help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="Torch device for model",
    )
    parser.add_argument(
        "--dont_draw_bbox", action="store_true", help="Avoid drawing bbox."
    )

    args = parser.parse_args()

    if args.save_path is not None and args.vqa_path is not None and len(args.save_path) > 0 and len(args.vqa_path) > 0:
        gt_dataset = VQADataset.load_dataset(args.vqa_path)

        save_path = Path(args.save_path)
        dataset_path = Path(args.dataset_path)

        model_args = create_args_for_qwen(
            model_path=args.model_path,
            temperature=0.1,  # Lower temperature for more consistent answers
            max_new_tokens=256
        )
        
        if not save_path.exists():
            pred_dataset = run_on_dataset(
                gt_dataset, model_args, dataset_path, save_path, args.batch_size, not args.dont_draw_bbox
            )
        else:
            pred_dataset = VQADataset.load_dataset(str(save_path))

        # Uncomment for evaluation
        # metric = MultipleChoiceMetric()
        # pprint.pprint(metric.evaluate_dataset(pred_dataset, gt_dataset))
        
    else:
        dataset_path = Path(args.dataset_path) 
        vqa_path = dataset_path / "generated_vqa_samples"
        output_folder = dataset_path / "model_outputs"
        
        model_suffix = "qwen-vl"
        if args.save_suffix is not None and len(args.save_suffix) > 0:
            model_suffix = model_suffix + args.save_suffix
        
        model_args = create_args_for_qwen(
            model_path=args.model_path,
            temperature=0.2,
            max_new_tokens=256
        )
        
        save_prefix = args.data_save_prefix if args.data_save_prefix is not None else ''
        
        for vqa_dataset_path in list(vqa_path.rglob('*.json')):
            if (save_prefix not in vqa_dataset_path.stem) or ("validation" not in vqa_dataset_path.stem):
                print(f'skipping {vqa_dataset_path.stem}')
                continue 
            
            print(f'processing {vqa_dataset_path.stem}')
            
            gt_dataset = VQADataset.load_dataset(str(vqa_dataset_path))

            save_path = output_folder / f'{vqa_dataset_path.stem}_{model_suffix}.json'
            metrics_save_path = save_path.with_name(f'{save_path.stem}_metrics.json')
            metric_latex_save_path = save_path.with_name(f'{save_path.stem}_tables.tex')

            if not save_path.exists():
                pred_dataset = run_on_dataset(
                    gt_dataset, model_args, dataset_path, save_path, 1, not args.dont_draw_bbox
                )
            else:
                pred_dataset = VQADataset.load_dataset(str(save_path))

            # Uncomment for evaluation
            # metric = MultipleChoiceMetric()
            # metric_results = metric.evaluate_dataset(pred_dataset, gt_dataset)
            # summarised_results = metric.summarise(metric_results)
            # summarised_results['model_name'] = model_suffix
            # 
            # with open(metrics_save_path, 'w') as f:
            #     json.dump(summarised_results, f)
            # 
            # metric.print_latex_tables(metric_results, model_suffix)
            # metric.save_latex_tables(metric_results, model_suffix, metric_latex_save_path)
            # print(f"Inference saved to {save_path}")
            # print(f"Latex tables saved to {metric_latex_save_path}")
        
if __name__ == "__main__":
    main()