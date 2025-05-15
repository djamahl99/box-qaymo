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
from waymovqa.questions.single_image_single_object import SingleImageSingleObjectQuestion
from waymovqa.answers.object_2d import Object2DAnswer
from waymovqa.answers.multi_object_2d import MultiObject2DAnswer
from waymovqa.metrics.coco import COCOEvaluator
from waymovqa.waymo_loader import WaymoDatasetLoader


class OWLv2Detector(nn.Module):
    def __init__(self, pretrained_name, class_names):
        super().__init__()

        self.pretrained_name = pretrained_name

        model = Owlv2ForObjectDetection.from_pretrained(self.pretrained_name)

        self.image_size = model.config.vision_config.image_size

        self.vision_model = model.owlv2.vision_model

        self.text_features = None

        if len(class_names) > 1:  # not placeholder (zero-shot inference)
            processor = AutoProcessor.from_pretrained(self.pretrained_name)
            inputs = processor(
                text=[[f"a {x}" for x in class_names]], return_tensors="pt"
            )
            self.text_features = model.owlv2.get_text_features(**inputs)

        self.image_processor = Owlv2ImageProcessor.from_pretrained(self.pretrained_name)

        self.class_head = model.class_head
        self.box_head = model.box_head
        self.objectness_head = model.objectness_head
        self.layer_norm = model.layer_norm
        self.sigmoid = nn.Sigmoid()

        self.vision_model.requires_grad_(False)
        self.class_head.requires_grad_(False)
        self.box_head.requires_grad_(False)
        self.layer_norm.requires_grad_(False)

    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        processor = AutoProcessor.from_pretrained(self.pretrained_name)
        model = Owlv2ForObjectDetection.from_pretrained(self.pretrained_name)

        inputs = processor(
            text=[texts], return_tensors="pt"
        )
        return model.owlv2.get_text_features(**inputs)

    def preprocess_image(self, image):
        return self.image_processor.preprocess(images=image, return_tensors="pt")

    def preprocess_batch(self, images):
        """
        Preprocess a batch of images for the model.

        Args:
            images: List of images to process (numpy arrays in RGB format)

        Returns:
            Dictionary containing preprocessed batch with pixel values tensor
        """
        if not isinstance(images, list):
            images = [images]

        # Process all images in a single batch
        return self.image_processor.preprocess(images=images, return_tensors="pt")

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.normalize_grid_corner_coordinates
    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # Computes normalized xy corner coordinates from feature_map.
        if not feature_map.ndim == 4:
            raise ValueError(
                "Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]"
            )

        device = feature_map.device
        num_patches = feature_map.shape[1]

        # TODO: Remove numpy usage.
        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)),
            axis=-1,
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1],
            box_coordinates.shape[2],
        )
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    def objectness_predictor(
        self, image_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        image_features = image_features.detach()
        objectness_logits = self.objectness_head(image_features)
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.compute_box_bias
    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(
            -box_coordinates + 1e-4
        )

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.box_predictor
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.class_predictor
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(
            image_feats, query_embeds, query_mask
        )

        return (pred_logits, image_class_embeds)

    def get_image_class_embeds(self, image_feats: torch.FloatTensor):
        image_class_embeds = self.class_head.dense0(image_feats)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )

        return image_class_embeds

    def post_process_boxes(self, pred_bboxes_xywh, target_size=[768, 768]):
        xy = pred_bboxes_xywh[..., 0:2]
        wh = pred_bboxes_xywh[..., 2:]

        xy1 = xy - wh * 0.5
        xy2 = xy + wh * 0.5
        xyxy = torch.cat((xy1, xy2), dim=-1)

        scale_fct = xyxy.new_tensor(
            [target_size[1], target_size[0], target_size[1], target_size[0]]
        ).reshape(1, 1, 4)
        # scale_fct = xyxy.new_tensor([1600, 900, 1600, 900]).reshape_as(xyxy)

        xyxy_scaled = xyxy * scale_fct

        return xyxy_scaled

    def forward_batch(self, pixel_values, target_size=[900, 1600]):
        """
        Process a batch of images efficiently.

        Args:
            pixel_values: Tensor of shape [batch_size, channels, height, width]
            target_size: Target size for the output boxes

        Returns:
            Dictionary containing predictions for the entire batch
        """
        batch_size = pixel_values.shape[0]

        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)

        with torch.no_grad():  # Disable gradient calculation for inference
            # Get vision model outputs
            last_hidden_state, pooled_output = self.vision_model(
                pixel_values, return_dict=False
            )

            # Process hidden states
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)

            # Resize class token for the whole batch at once
            new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
            class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

            # Merge image embedding with class tokens
            image_embeds = image_embeds[:, 1:, :] * class_token_out
            image_embeds = self.layer_norm(image_embeds)

            # Reshape to [batch_size, num_patches, num_patches, hidden_size]
            patch_size = int(np.sqrt(image_embeds.shape[1]))
            new_size = (batch_size, patch_size, patch_size, image_embeds.shape[-1])
            feature_map = image_embeds.reshape(new_size)

            # Reshape for further processing
            image_feats = feature_map.reshape(
                batch_size, patch_size * patch_size, feature_map.shape[-1]
            )

            # Run all predictions in parallel
            pred_boxes = self.box_predictor(image_feats, feature_map)
            pred_boxes = self.post_process_boxes(pred_boxes, target_size)

            objectness_logits = self.objectness_predictor(image_feats)
            image_class_embeds = self.get_image_class_embeds(image_feats)

            print(
                "image_class_embeds, image_feats",
                image_class_embeds.shape,
                image_feats.shape,
            )

            # Return all outputs in a dictionary
            return {
                "pred_boxes": pred_boxes,
                "image_feats": image_feats,
                "objectness_logits": objectness_logits,
                "image_class_embeds": image_class_embeds,
            }

    def forward(self, images, target_size=[900, 1600]):
        # Embed images
        last_hidden_state, pooled_output = self.vision_model(images, return_dict=False)

        image_embeds = self.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        # changed name from image_text_embedder
        feature_map = image_embeds

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(
            feature_map, (batch_size, num_patches * num_patches, hidden_dim)
        )

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        # max_text_queries = input_ids.shape[0] // batch_size
        # query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        # input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        # query_mask = input_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        # (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)
        pred_boxes = self.post_process_boxes(pred_boxes, target_size)

        # Predict objectness
        objectness_logits = self.objectness_predictor(image_feats)

        # image class embeddings
        image_class_embeds = self.get_image_class_embeds(image_feats)

        # return pred_boxes, image_feats, objectness_logits
        return dict(
            pred_boxes=pred_boxes,
            image_feats=image_feats,
            objectness_logits=objectness_logits,
            image_class_embeds=image_class_embeds,
        )

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
            final_scores = (prompt_out + selected_objectness.reshape(prompt_out.shape)) / 2.0
            # final_scores = (prompt_out * selected_objectness.reshape(prompt_out.shape))
            # final_scores = (prompt_out * selected_objectness.reshape(prompt_out.shape))

            print('final_scores', final_scores.shape)
            print('final_scores', final_scores.min(), final_scores.max())


            final_scores = final_scores.reshape(num_boxes, 1)
            final_boxes = selected_boxes.reshape(num_boxes, 4)

            final_thresh = max(conf_thresh, final_scores.min()) # at least one detection (min)
            # indices = torch.arange(num_boxes)[final_scores >= final_thresh]
            indices = np.nonzero(final_scores >= final_thresh)[0]
            # indices = np.argmax(final_scores, axis=0).reshape(-1)

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

            # img_vis = cv2.cvtColor(current_batch_loaded[i], cv2.COLOR_RGB2BGR)

            # import matplotlib.pyplot as plt

            # cmap = plt.get_cmap('jet')
            
            # for bidx, (box, conf) in enumerate(zip(final_boxes, final_scores)):
            #     x1, y1, x2, y2 = map(int, box)
            #     color = tuple(int(x*255) for x in cmap(bidx / len(final_boxes)))
            #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)
                
            #     print('conf', conf)
            #     # Add confidence text
            #     text = f'{conf[0]:.2f}'
            #     cv2.putText(
            #         img_vis, text, (x1, y1), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            #     )

            # cv2.imwrite(f'vis_{i}.jpg', img_vis)

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
        assert isinstance(question, SingleImageSingleObjectQuestion) and isinstance(
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
    dataset_path = Path(args.dataset_path)

    if not save_path.exists():
        owlvit_model = OWLv2Detector(args.pretrained_name, ['placeholder']).to(args.device)

        pred_dataset = run_on_dataset(gt_dataset, owlvit_model, dataset_path, save_path, args.batch_size, args.conf_thresh, args.nms_thresh)
    else:
        pred_dataset = VQADataset.load_dataset(str(save_path))

    # evaluate?
    coco_eval = COCOEvaluator(dataset_path)

    predictions = [x[1] for x in pred_dataset.samples]
    ground_truths = [x[1] for x in gt_dataset.samples]

    # get the questions from both pred / gt to check they are matching
    prompts0 = [x[0].question for x in pred_dataset.samples]
    prompts1 = [x[0].question for x in gt_dataset.samples]
    questions = [x[0] for x in gt_dataset.samples]

    print('prompts0', len(prompts0))
    print('prompts1', len(prompts1))
    assert all([prompts0[i] == prompts1[i] for i in range(len(prompts0))]) and len(prompts0) == len(prompts1)

    pprint.pprint(coco_eval.evaluate(predictions, ground_truths, questions))

if __name__ == "__main__":
    main()
