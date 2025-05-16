import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Tuple, Optional, List
import torch
from pathlib import Path
from PIL import Image
import pprint
import re


from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.questions import *
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.metrics.multiple_choice import MultipleChoiceMetric

from transformers import AutoTokenizer
import torch
from PIL import Image

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def generate_responses_for_inputs(text_strs, image_paths, args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    responses = []

    for text, image_path in zip(text_strs, image_paths):
        
        inp = text
        if image_path is not None:
            image = load_image(image_path)
            image_tensor = (
                image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                .half()
                .cuda()
            )

            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], inp)

        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        responses.append(outputs)

    return responses


def run_on_dataset(
    gt_dataset: VQADataset,
    args, 
    dataset_path: Path,
    save_path: Path,
    batch_size=1,
) -> VQADataset:
    """
    Processes a dataset with batch processing.
    Args:
        gt_dataset: VQADataset to run the model on
        args: Arguments for llava
        dataset_path: path to extracted data
        save_path: path to save results
        batch_size: Maximum number of images to process in a single batch
    """
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    pred_dataset = VQADataset(tag=f"llava_{save_path.stem}")

    # Prepare batches for processing
    batch_text_strs = []
    batch_image_paths = []
    batch_questions = []
    batch_gt_answers = []

    for sample in gt_dataset.samples:
        question = sample.question
        gt_answer = sample.answer

        image_path = str(dataset_path / question.image_path)

        if isinstance(question, SingleImageMultipleChoiceQuestion):
            text = question.question
            # choices_text = "\n".join(
            #     [f"{i+1}. {choice}" for i, choice in enumerate(question.choices)]
            # )
            # prompt = f"{text}\nChoose the most appropriate answer from the following options:\n{choices_text}"

            prompt = f'{question.question}\nRespond with only the full name of your selection (e.g., "{question.choices[0]}")..'

            batch_text_strs.append(prompt)
            batch_image_paths.append(image_path)
            batch_questions.append(question)
            batch_gt_answers.append(gt_answer)

        elif isinstance(question, SingleImageMultiplePromptQuestion):
            # For multiple prompt questions, add each prompt separately
            for prompt in question.prompts:
                batch_text_strs.append(prompt)
                batch_image_paths.append(image_path)
                batch_questions.append(question)
                batch_gt_answers.append(gt_answer)
        else:
            raise TypeError(f"Question type not valid for this model")

        # Process batch if it reaches the specified size
        if len(batch_text_strs) >= batch_size:
            process_batch(
                batch_text_strs,
                batch_image_paths,
                batch_questions,
                image_processor,
                model,
                conv_templates[args.conv_mode],
                tokenizer,
                pred_dataset,
            )
            batch_text_strs = []
            batch_image_paths = []
            batch_questions = []
            batch_gt_answers = []

    # Process any remaining samples
    if batch_text_strs:
        process_batch(
            batch_text_strs,
            batch_image_paths,
            batch_questions,
            image_processor,
            model,
            conv_templates[args.conv_mode],
            tokenizer,
            pred_dataset,
        )

    # Save results
    pred_dataset.save_dataset(save_path)
    return pred_dataset


def process_batch(
    batch_text_strs,
    batch_image_paths,
    batch_questions,
    image_processor,
    model,
    conv_template,  # Changed from conv to conv_template
    tokenizer,
    pred_dataset,
):
    """Helper function to process a batch of inputs using proper batching"""
    # Load all images in batch
    images = [load_image(path) if path is not None else None for path in batch_image_paths]
    image_tensors = torch.cat([
        image_processor.preprocess(img, return_tensors="pt")["pixel_values"].half()
        for img in images if img is not None
    ]).cuda()
    
    # Prepare all inputs in batch
    input_ids_list = []
    image_indices = []  # To track which image goes with which input
    
    for i, (text, image_path) in enumerate(zip(batch_text_strs, batch_image_paths)):
        # Create a fresh conversation for each input to avoid contamination
        conv = conv_template.copy()
        
        inp = text
        if image_path is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            
            conv.append_message(conv.roles[0], inp)
            image_indices.append(i)  # Record which image this prompt uses
        else:
            conv.append_message(conv.roles[0], inp)
        
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        
        input_ids_list.append(input_ids)
    
    # Process all inputs in one batch
    responses = []
    for i, input_ids in enumerate(input_ids_list):
        # Create fresh conversation to get stop strings
        conv = conv_template.copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Get the correct image tensor for this input
        img_tensor = image_tensors[image_indices.index(i)].unsqueeze(0) if i in image_indices else None
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=img_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        
        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(batch_text_strs[i])
        print(f"OUTPUT:", outputs)
        with open(f'raw_output_{i}.txt', 'w') as f:
            f.write(outputs)

        responses.append(outputs)

    # The rest of your code for processing responses remains the same
    # Group responses by original question (for multiple prompt questions)
    response_dict = {}
    for i, response in enumerate(responses):
        question = batch_questions[i]
        question_id = id(question)  # Use object id as a unique identifier

        if question_id not in response_dict:
            response_dict[question_id] = {"question": question, "responses": []}

        response_dict[question_id]["responses"].append(response)

    # Process each question's responses
    for question_data in response_dict.values():
        question = question_data["question"]
        responses = question_data["responses"]

        if isinstance(question, SingleImageMultipleChoiceQuestion):
            # Only one response for single prompt questions
            answer = parse_multiple_choice_response(responses[0], question.choices)
            answer = MultipleChoiceAnswer(answer=answer, choices=question.choices)

        elif isinstance(question, SingleImageMultiplePromptQuestion):
            # Multiple responses for multiple prompt questions
            answer = determine_best_answer_from_prompts(responses, question.choices)
            answer = MultipleChoiceAnswer(answer=answer, choices=question.choices)

        pred_dataset.add_sample(question, answer)


def parse_multiple_choice_response(response: str, choices: List[str]) -> str:
    """
    Parse the LLaVA response to extract the chosen answer for multiple choice questions.
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
    sorted_choices = sorted(range(len(choices)), key=lambda i: len(normalized_choices[i]), reverse=True)
    for i in sorted_choices:
        if normalized_choices[i] in response:
            return choices[i]
    
    # Fallback to similarity for truly ambiguous cases
    if compute_text_similarity:  # Assuming this function exists
        return max(choices, key=lambda x: compute_text_similarity(x.lower(), response))
    
    return "AMBIGUOUS_RESPONSE"


def determine_best_answer_from_prompts(responses: List[str], choices: List[str]) -> str:
    """
    Determine the best answer from multiple prompt responses.
    """
    # Simple implementation: count which choice gets the most support across responses
    vote_counts = {choice: 0 for choice in choices}

    for response in responses:
        for choice in choices:
            if choice.lower() in response.lower():
                vote_counts[choice] += 1

    # Return the choice with the most votes
    return max(vote_counts.items(), key=lambda x: x[1])[0]


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple text similarity between two strings.
    """
    # This is a simple implementation - you might want to use a more sophisticated approach
    text1_tokens = set(text1.lower().split())
    text2_tokens = set(text2.lower().split())
    intersection = text1_tokens.intersection(text2_tokens)
    union = text1_tokens.union(text2_tokens)
    return len(intersection) / len(union) if union else 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LLAVA on VQA questions")
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
        "--model_path",
        type=str,
        required=False,
        default="llava_llama_2",
        help="Pretrained model name",
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
    dataset_path = Path(args.dataset_path)

    llava_args = argparse.Namespace(
        model_path="liuhaotian/llava-v1.5-7b",  # Specify the correct model path
        model_base=None,
        image_file=None,  # Not needed since we're passing image paths separately
        num_gpus=1,
        conv_mode=None,
        temperature=0.2,
        max_new_tokens=512,
        load_8bit=False,
        load_4bit=False,
        debug=False,
    )

    if not save_path.exists():
        pred_dataset = run_on_dataset(
            gt_dataset, llava_args, dataset_path, save_path, args.batch_size
        )
    else:
        pred_dataset = VQADataset.load_dataset(str(save_path))

    # evaluate?
    metric = MultipleChoiceMetric()

    pprint.pprint(metric.evaluate_dataset(pred_dataset, gt_dataset))

    # predictions = [x.answer for x in pred_dataset.samples]
    # ground_truths = [x.answer for x in gt_dataset.samples]

    # # get the questions from both pred / gt to check they are matching
    # questions0 = [x.question for x in pred_dataset.samples]
    # questions1 = [x.question for x in gt_dataset.samples]
    # questions = questions0

    # assert all([questions0[i] == questions1[i] for i in range(len(questions0))]) and len(
    #     questions0
    # ) == len(questions1)

    # pprint.pprint(metric.evaluate(predictions, ground_truths, questions))


if __name__ == "__main__":
    main()
