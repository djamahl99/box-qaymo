import sys
import os
import json
from tqdm import tqdm
import random
from typing import Dict, Tuple, Optional, List
import torch
from pathlib import Path
from PIL import Image
import pprint
import re
from dataclasses import dataclass, field
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.questions import *
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token


@dataclass
class TrainingConfig:
    """Training configuration"""
    dataset_path: str
    vqa_path: str
    output_dir: str = "./llava_finetuned"
    model_name: str = "liuhaotian/llava-v1.5-7b"
    train_split: float = 0.8
    batch_size: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 100
    conv_mode: str = "llava_v1"
    load_4bit: bool = False  # Set to False to avoid quantization issues
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VQATrainingDataset(Dataset):
    """Simplified dataset for training LLaVA on VQA tasks"""
    
    def __init__(
        self, 
        vqa_dataset: VQADataset, 
        dataset_path: Path,
        tokenizer,
        image_processor,
        conv_mode: str = "llava_v1",
        max_length: int = 512
    ):
        self.vqa_dataset = vqa_dataset
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template = conv_templates[conv_mode]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.vqa_dataset.samples)
    
    def load_image(self, image_path, bbox=None):
        """Load and process image, optionally with bounding box"""
        img_vis = cv2.imread(str(self.dataset_path / image_path))
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 6)

        return Image.fromarray(img_vis)
    
    def __getitem__(self, idx):
        sample = self.vqa_dataset.samples[idx]
        question = sample.question
        answer = sample.answer
        
        # Handle different question types (simplified)
        if isinstance(question, (SingleImageMultipleChoiceQuestion, SingleImageQuestion)):
            image_path = question.image_path
            text_question = question.question
            
            if hasattr(question, 'choices'):
                choices = question.choices
            elif isinstance(answer, MultipleChoiceAnswer):
                choices = answer.choices
            else:
                raise ValueError("No choices found for multiple choice question")
                
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
                raise ValueError("No choices found for multiple choice question")
        else:
            raise TypeError(f"Unsupported question type: {type(question)}")
        
        # Get bounding box if available
        bbox = None
        if isinstance(question.data, dict) and "bbox" in question.data:
            bbox = question.data['bbox']
        
        # Load and process image
        try:
            image = self.load_image(image_path, bbox)
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image tensor
            image_tensor = torch.zeros(3, 224, 224)
        
        # Prepare conversation
        conv = self.conv_template.copy()
        
        # Create prompt with choices
        prompt = f'{text_question}\nRespond with only the full name of your selection (e.g., "{random.choice(choices)}").'
        
        # Add image token
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        
        # Get the correct answer
        if isinstance(answer, MultipleChoiceAnswer):
            target_answer = answer.answer
        else:
            target_answer = str(answer)
            
        conv.append_message(conv.roles[1], target_answer)
        conversation = conv.get_prompt()
        
        # Tokenize
        try:
            input_ids = tokenizer_image_token(
                conversation, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Error tokenizing conversation: {e}")
            # Return a dummy sample
            return {
                "input_ids": torch.tensor([1]),
                "labels": torch.tensor([1]),
                "images": torch.zeros(3, 224, 224),
                "attention_mask": torch.tensor([1])
            }
        
        # Create labels (same as input_ids but with prompt tokens masked)
        targets = input_ids.clone()
        
        # Find where the assistant's response starts
        sep = conv.sep + conv.roles[1] + ": "
        parts = conversation.split(sep)
        if len(parts) >= 2:
            parts[0] += sep
            
            # Tokenize the prompt part to find where to start unmasking
            try:
                prompt_tokens = tokenizer_image_token(
                    parts[0],
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt"
                )
                
                # Mask prompt tokens in targets
                targets[:len(prompt_tokens)] = IGNORE_INDEX
            except:
                # If tokenization fails, mask first half of tokens
                targets[:len(targets)//2] = IGNORE_INDEX
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            targets = targets[:self.max_length]
        
        return {
            "input_ids": input_ids.flatten(),
            "labels": targets.flatten(),
            "images": image_tensor,
            "attention_mask": torch.ones_like(input_ids.flatten())
        }


def collate_fn(batch):
    """Custom collate function"""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Get max length in batch
    max_len = max([len(item["input_ids"]) for item in batch])
    
    input_ids = []
    labels = []
    attention_masks = []
    images = []
    
    for item in batch:
        # Pad sequences
        input_id = item["input_ids"]
        label = item["labels"]
        attention_mask = item["attention_mask"]
        
        padding_length = max_len - len(input_id)
        
        if padding_length > 0:
            # Pad from the left
            input_id = torch.cat([torch.zeros(padding_length, dtype=input_id.dtype), input_id])
            attention_mask = torch.cat([torch.zeros(padding_length, dtype=attention_mask.dtype), attention_mask])
            label = torch.cat([torch.full((padding_length,), IGNORE_INDEX, dtype=label.dtype), label])
        
        input_ids.append(input_id)
        labels.append(label)
        attention_masks.append(attention_mask)
        images.append(item["images"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks),
        "images": torch.stack(images)
    }


def train_model(config: TrainingConfig):
    """Main training function"""
    
    # Load dataset
    print("Loading VQA dataset...")
    vqa_dataset = VQADataset.load_dataset(config.vqa_path)
    
    # Split dataset
    total_samples = len(vqa_dataset.samples)
    train_size = int(total_samples * config.train_split)
    
    # Shuffle samples before splitting
    random.shuffle(vqa_dataset.samples)
    
    train_samples = vqa_dataset.samples[:train_size]
    eval_samples = vqa_dataset.samples[train_size:]
    
    train_dataset = VQADataset(tag=f"{vqa_dataset.tag}_train")
    eval_dataset = VQADataset(tag=f"{vqa_dataset.tag}_eval")
    
    train_dataset.samples = train_samples
    eval_dataset.samples = eval_samples
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    
    # Load model
    print("Loading LLaVA model...")
    disable_torch_init()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        config.model_name,
        None,  # model_base
        "llava-v1.5-7b",  # model name
        False,  # load_8bit
        config.load_4bit
    )
    
    # Move model to device
    model = model.to(config.device)
    
    # Enable training mode
    model.train()
    
    # Enable gradient computation for vision tower and language model
    for param in model.parameters():
        param.requires_grad = True
    
    # Create datasets
    print("Creating training datasets...")
    train_torch_dataset = VQATrainingDataset(
        train_dataset,
        Path(config.dataset_path),
        tokenizer,
        image_processor,
        config.conv_mode
    )
    
    eval_torch_dataset = VQATrainingDataset(
        eval_dataset,
        Path(config.dataset_path),
        tokenizer,
        image_processor,
        config.conv_mode
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    eval_loader = DataLoader(
        eval_torch_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Setup learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - config.warmup_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_steps]
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            if batch is None:
                continue
                
            # Move batch to device
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            images = batch["images"].to(config.device)
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images
                )
                loss = outputs.loss
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr:.2e}")
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                save_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                save_path.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'config': config
                }, save_path / "pytorch_model.bin")
                
                print(f"Checkpoint saved to {save_path}")
    
    # Save final model
    final_save_path = Path(config.output_dir) / "final_model"
    final_save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_save_path / "pytorch_model.bin")
    
    # Also save the tokenizer
    tokenizer.save_pretrained(final_save_path)
    
    print(f"Training completed! Final model saved to {final_save_path}")
    
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA on VQA dataset (minimal version)")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed waymo dataset")
    parser.add_argument("--vqa_path", type=str, required=True, help="Path to vqa gt dataset")
    parser.add_argument("--output_dir", type=str, default="./llava_finetuned")
    parser.add_argument("--model_name", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        dataset_path=args.dataset_path,
        vqa_path=args.vqa_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_split=args.train_split,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        conv_mode=args.conv_mode
    )
    
    # Train model
    model = train_model(config)
    
    print("Training completed!")


if __name__ == "__main__":
    main()