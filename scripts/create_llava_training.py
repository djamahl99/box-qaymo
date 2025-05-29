import json
import random
from pathlib import Path
from typing import List, Dict, Any

from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.questions import *
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.questions.multi_image_multi_choice import MultipleImageMultipleChoiceQuestion
from waymovqa.questions.multi_image import MultipleImageQuestion

def convert_vqa_to_llava_format(vqa_dataset, output_path: str):
    """
    Convert your VQA dataset to LLaVA's expected JSON format.
    
    LLaVA expects:
    [
        {
            "id": "unique_id",
            "image": "relative/path/to/image.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat is in this image?"
                },
                {
                    "from": "gpt", 
                    "value": "This is a car."
                }
            ]
        },
        ...
    ]
    """
    
    llava_data = []
    
    for idx, sample in enumerate(vqa_dataset.samples):
        question = sample.question
        answer = sample.answer
        
        # Extract image path and question text based on question type
        if isinstance(question, (SingleImageMultipleChoiceQuestion, SingleImageQuestion)):
            image_path = question.image_path
            text_question = question.question
            
            if hasattr(question, 'choices'):
                choices = question.choices
            elif isinstance(answer, MultipleChoiceAnswer):
                choices = answer.choices
            else:
                choices = []
                
        elif isinstance(question, (MultipleImageMultipleChoiceQuestion, MultipleImageQuestion)):
            # Use the first image or best camera
            best_camera_name = "FRONT"
            if isinstance(question.data, dict) and "best_camera_name" in question.data:
                best_camera_name = question.data['best_camera_name']
                
            cam_idx = next((i for i, cam_name in enumerate(question.camera_names) if cam_name == best_camera_name), 0)
            image_path = question.image_paths[cam_idx]
            text_question = question.question
            
            if hasattr(question, 'choices'):
                choices = question.choices
            elif isinstance(answer, MultipleChoiceAnswer):
                choices = answer.choices
            else:
                choices = []
        else:
            print(f"Skipping unsupported question type: {type(question)}")
            continue
        
        # Get the correct answer
        if isinstance(answer, MultipleChoiceAnswer):
            target_answer = answer.answer
        else:
            target_answer = str(answer)
        
        # Format the question with choices if it's multiple choice
        if choices:
            formatted_question = f'{text_question}\nRespond with only the full name of your selection (e.g., "{random.choice(choices)}").'
        else:
            formatted_question = text_question
        
        # Create LLaVA format entry
        llava_entry = {
            "id": question.question_id,
            "image": f"{Path(image_path).name}",  # Relative path
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{formatted_question}"
                },
                {
                    "from": "gpt",
                    "value": target_answer
                }
            ]
        }
        
        llava_data.append(llava_entry)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"Converted {len(llava_data)} samples to LLaVA format: {output_path}")
    return llava_data

def create_training_script(output_dir: str, data_json_path: str, image_folder: str, model_name: str = "liuhaotian/llava-v1.5-7b"):
    """
    Create a training script that uses the original LLaVA training code with your data.
    """
    
    script_content = f'''#!/bin/bash

# Training script for VQA dataset using LLaVA

export WANDB_PROJECT="waymo-vqa-finetuning-llava-v1.5-7b-full"

python llava/train/train_mem.py \\
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \\
    --model_name_or_path {model_name} \\
    --version v1 \\
    --data_path {data_json_path} \\
    --image_folder {image_folder} \\
    --vision_tower openai/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --group_by_modality_length True \\
    --bf16 True \\
    --output_dir {output_dir} \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 8 \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --evaluation_strategy "no" \\
    --save_strategy "steps" \\
    --save_steps 50000 \\
    --save_total_limit 1 \\
    --learning_rate 2e-4 \\
    --weight_decay 0. \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 4 \\
    --lazy_preprocess True \\
    --report_to wandb \\
    --bits 16 \\
    --run_name "waymo-vqa-llava-v1-$(date +%Y%m%d_%H%M%S)" 
'''
    
    script_path = f"{output_dir}/train_vqa.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    import os
    os.chmod(script_path, 0o755)
    
    print(f"Training script created: {script_path}")
    return script_path

# Example usage function
def setup_llava_training(vqa_dataset, base_output_dir: str, dataset_path: str):
    """
    Complete setup for training LLaVA on your VQA dataset.
    
    Args:
        vqa_dataset: Your VQA dataset object
        base_output_dir: Where to save training outputs
        dataset_path: Path to your dataset (contains camera_images folder)
    """
    
    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert dataset to LLaVA format
    data_json_path = output_dir / "vqa_llava_format.json"
    convert_vqa_to_llava_format(
        vqa_dataset, 
        str(data_json_path),
    )
    
    # 2. Create training script
    image_folder = str(Path(dataset_path) / "camera_images")  # Absolute path to dataset
    training_output_dir = output_dir / "checkpoints"
    
    training_output_dir.mkdir(exist_ok=True)
    
    script_path = create_training_script(
        str(training_output_dir),
        str(data_json_path),
        image_folder
    )
    
    print("\\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print(f"1. Dataset converted: {data_json_path}")
    print(f"2. Training script: {script_path}")
    print(f"3. Image folder: {image_folder}")
    print(f"4. Output directory: {training_output_dir}")
    print("\\nTo start training, run:")
    print(f"   bash {script_path}")
    print("="*50)
    
    return {
        "data_json": str(data_json_path),
        "script_path": script_path,
        "image_folder": image_folder,
        "output_dir": str(training_output_dir)
    }

# Alternative: Minimal modification to use existing training code
def create_minimal_training_config():
    """
    Shows how to minimally modify the existing LLaVA training arguments.
    """
    
    config_example = '''
# Minimal changes to use your VQA dataset with existing LLaVA training:

# 1. Set these arguments in your training call:
training_args = TrainingArguments(
    # ... other args ...
    lora_enable=True,           # Enable LoRA
    lora_r=128,                 # LoRA rank
    lora_alpha=256,             # LoRA alpha
    lora_dropout=0.05,          # LoRA dropout
    bits=16,                    # Use 16-bit (or 4 for 4-bit quantization)
)

data_args = DataArguments(
    data_path="path/to/your/vqa_llava_format.json",    # Your converted JSON
    image_folder="path/to/your/dataset",               # Folder containing camera_images/
    is_multimodal=True,
    image_aspect_ratio='pad'    # or 'square'
)

model_args = ModelArguments(
    model_name_or_path="liuhaotian/llava-v1.5-7b",
    version="v1",               # Use v1 conversation format
    vision_tower="openai/clip-vit-large-patch14-336",
    mm_projector_type="mlp2x_gelu",
    mm_vision_select_layer=-2,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
)

# 2. Then just call the existing train() function!
# The existing LazySupervisedDataset will handle your converted JSON data.
'''
    
    return config_example

def main():
    dataset_path = Path("/home/uqdetche/waymo_vqa_dataset")
    generated_samples_path = dataset_path / "generated_vqa_samples"
    ds = VQADataset(tag="training_all")
    
    random.seed(42)
    
    save_prefix = "26_05_2025_export"
    
    save_path = generated_samples_path / f'{save_prefix}_traininig.json'
    
    for path in generated_samples_path.rglob(f'{save_prefix}*.json'):
        if ("training" not in path.name) or ("balance_stats" in path.name):
            continue

        ds1 = VQADataset.load_dataset(str(path))
        
        for sample in ds1.samples:
            ds.add_sample(sample.question, sample.answer)
            
    ds.save_dataset(str(save_path))
    print(f'saved to {save_path}')
    
    output_dir = "/scratch/user/uqdetche/llava_training"
    setup_llava_training(ds, output_dir, str(dataset_path))

if __name__ == "__main__":
    main()