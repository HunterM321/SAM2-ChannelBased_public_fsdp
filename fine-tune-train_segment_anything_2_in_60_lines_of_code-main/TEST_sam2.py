# TEST.py

import os
import cv2
import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from peft import LoraConfig, get_peft_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Importing Florence-2 related libraries
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import Dice
from torchmetrics.classification import BinaryJaccardIndex

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load the YAML configuration file.

    Args:
        config_path (str, optional): Path to the config file. Defaults to "config.yaml".

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_output_directory(directory: str):
    """Create a directory for saving test results."""
    os.makedirs(directory, exist_ok=True)


def load_florence2_model(model_id: str):
    """
    Load the Florence-2 model for object detection.

    Args:
        model_id (str): Path to the Florence-2 model checkpoint.

    Returns:
        model: Loaded Florence-2 model.
        processor: Corresponding processor.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def florence2(task_prompt: str, image: Image.Image, processor, model, text_input: Optional[str] = None) -> Dict:
    """
    Calling the Microsoft Florence2 model.

    Args:
        task_prompt (str): Task prompt for Florence-2.
        image (PIL.Image.Image): Input image.
        processor: Florence-2 processor.
        model: Loaded Florence-2 model.
        text_input (str, optional): Additional text input. Defaults to None.

    Returns:
        dict: Parsed answer containing bounding boxes and labels.
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))
    
    return parsed_answer


def plot_bbox(image: Image.Image, data: Dict):
    """
    Plot bounding boxes on the image for visualization.

    Args:
        image (PIL.Image.Image): Input image.
        data (dict): Dictionary containing bounding boxes and labels.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    plt.show()


def load_test_data(csv_file: str, data_dir: str, sample_size: int = 10, random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Load and sample test data from a CSV file.

    Args:
        csv_file (str): Path to the test CSV file.
        data_dir (str): Directory containing test images and masks.
        sample_size (int, optional): Number of samples to draw. Defaults to 10.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        list of dict: Sampled image and annotation paths.
    """
    test_df = pd.read_csv(csv_file)
    sampled_test_df = test_df.sample(n=sample_size, random_state=random_state)

    sampled_data = []
    for _, row in sampled_test_df.iterrows():
        filename = row['file_name']
        image_path = os.path.join(data_dir, "images", filename)
        annotation_path = os.path.join(data_dir, "masks", filename)
        sampled_data.append({"image": image_path, "annotation": annotation_path})

    return sampled_data


def configure_sam2_model(model_cfg: str, checkpoint_path: str, lora_checkpoint: str, device: str = "cuda") -> torch.nn.Module:
    """
    Build and configure the SAM2 model with LoRA.

    Args:
        model_cfg (str): Path to the model configuration file.
        checkpoint_path (str): Path to the trained SAM2 model checkpoint.
        lora_checkpoint (str): Path to the LoRA checkpoint.
        device (str, optional): Device to load the model on. Defaults to "cuda".

    Returns:
        torch.nn.Module: Configured model.
    """
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=[
            "sam_mask_decoder.transformer.layers.0.self_attn.q_proj",
            "sam_mask_decoder.transformer.layers.0.self_attn.k_proj",
            "sam_mask_decoder.transformer.layers.0.self_attn.v_proj",
            "sam_mask_decoder.transformer.layers.0.self_attn.out_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.out_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.k_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.out_proj",
            "memory_attention.layers.0.self_attn.q_proj",
            "memory_attention.layers.0.self_attn.k_proj",
            "memory_attention.layers.0.self_attn.v_proj",
            "memory_attention.layers.0.self_attn.out_proj",
            "memory_attention.layers.0.cross_attn_image.q_proj",
            "memory_attention.layers.0.cross_attn_image.k_proj",
            "memory_attention.layers.0.cross_attn_image.v_proj",
            "memory_attention.layers.0.cross_attn_image.out_proj",
        ],
        lora_dropout=0.0,
        bias="none",
    )

    # Apply the LoRA configuration to the model
    model = get_peft_model(sam2_model, lora_config)

    # Load LoRA checkpoint
    if os.path.exists(lora_checkpoint):
        lora_state_dict = torch.load(lora_checkpoint)
        model.load_state_dict(lora_state_dict, strict=False)
    else:
        raise FileNotFoundError(f"LoRA checkpoint not found at: {lora_checkpoint}")

    return model


def calculate_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between ground truth and prediction masks.

    Args:
        gt_mask (np.ndarray): Ground truth mask.
        pred_mask (np.ndarray): Predicted mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def calculate_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient between ground truth and prediction masks.

    Args:
        gt_mask (np.ndarray): Ground truth mask.
        pred_mask (np.ndarray): Predicted mask.

    Returns:
        float: Dice score.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    return (2.0 * intersection) / (gt_mask.sum() + pred_mask.sum()) if (gt_mask.sum() + pred_mask.sum()) > 0 else 0.0


def get_bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    """
    Extract bounding box from the mask.

    Args:
        mask (np.ndarray): Annotation mask.

    Returns:
        list or None: Bounding box in [x, y, width, height] format or None if no valid bbox is found.
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None  # Return None if the mask is empty or has no positive values

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]


def add_label(image: np.ndarray, label: str, position: tuple, font_scale: float = 0.8,
              thickness: int = 2, color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Add a text label to the image at the specified position.

    Args:
        image (np.ndarray): Image to annotate.
        label (str): Text label.
        position (tuple): (x, y) coordinates for the label.
        font_scale (float, optional): Font size. Defaults to 0.8.
        thickness (int, optional): Text thickness. Defaults to 2.
        color (tuple, optional): Text color in BGR. Defaults to white.

    Returns:
        np.ndarray: Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Binarize the mask by keeping only the highest label.

    Args:
        mask (np.ndarray): Input mask.

    Returns:
        np.ndarray: Binarized mask.
    """
    highest_label = np.max(mask)
    return (mask == highest_label).astype(np.uint8)


def load_annotation(annotation_path: str, chamber: str) -> np.ndarray:
    """
    Load and process the annotation mask based on the chamber type.

    Args:
        annotation_path (str): Path to the annotation mask.
        chamber (str): Chamber type ('LV', 'LA', or others).

    Returns:
        np.ndarray: Processed annotation mask.
    """
    annotation = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
    annotation = cv2.resize(annotation, (128, 128))
    if annotation is None:
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    if chamber == "LV":
        mask = annotation[..., 0]
    elif chamber == "LA":
        mask = annotation[..., 1]
    else:
        mask = annotation[..., 2]

    return mask


def evaluate_model(sampled_data: List[Dict[str, Any]], predictor: SAM2ImagePredictor, chamber: str,
                  output_dir: str, use_whole_image: bool = True, florence_model: Optional[Any] = None,
                  florence_processor: Optional[Any] = None, task_prompt: str = '<OD>') -> None:
    """
    Evaluate the model on sampled data and save visualizations.

    Args:
        sampled_data (list): List of dictionaries with image and annotation paths.
        predictor (SAM2ImagePredictor): Image predictor instance.
        chamber (str): Chamber type.
        output_dir (str): Directory to save the results.
        use_whole_image (bool, optional): Whether to use the whole image for bounding box. Defaults to True.
        florence_model (Any, optional): Loaded Florence-2 model. Required if use_whole_image is False.
        florence_processor (Any, optional): Loaded Florence-2 processor. Required if use_whole_image is False.
        task_prompt (str, optional): Task prompt for Florence-2. Defaults to '<OD>'.
    """

    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    iou_scores = []
    dice_scores = []
    predictor.model.eval()

    for i, sample in enumerate(tqdm(sampled_data, desc="Evaluating samples")):
        image_path = sample["image"]
        annotation_path = sample["annotation"]

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        try:
            # Load and process annotation
            annotation = load_annotation(annotation_path, chamber)
        except FileNotFoundError:
            continue

        # Determine bounding box
        if use_whole_image:
            img_height, img_width = annotation.shape[:2]
            bbox = [0, 0, img_width, img_height]
        else:
            if florence_model is None or florence_processor is None:
                raise ValueError("Florence-2 model and processor must be provided when use_whole_image is False.")

            # Obtain bounding boxes from Florence-2
            od_results = florence2(task_prompt, pil_image, florence_processor, florence_model)
            if '<OD>' not in od_results:
                continue  # Skip if no object detection results
            data = od_results['<OD>']
            if not data['bboxes']:
                continue  # Skip if no bounding boxes detected
            # For simplicity, use the first detected bounding box
            bbox = data['bboxes'][0]
            # Optionally, you can handle multiple bounding boxes as needed

        if bbox is None:
            continue  # Skip if no valid bounding box is found

        # Binarize annotation mask
        binary_annotation = binarize_mask(annotation)

        # Set image for predictor and predict masks
        predictor.set_image(image_rgb)
        prd_masks, scores, _ = predictor.predict(box=bbox)

        if prd_masks.size == 0:
            continue

        # Sort masks by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        sorted_masks = prd_masks[sorted_indices].astype(bool)

        # Select the highest scoring mask
        pred_mask = sorted_masks[0].astype(np.uint8)

        # Resize the prediction
        pred_mask = cv2.resize(pred_mask, (128, 128))

        # Compute various matrics
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matric_pre = BinaryPrecision().to(device)
        metric_rec = BinaryRecall().to(device)
        metric_acc = BinaryAccuracy().to(device)
        metric_dic = Dice().to(device)
        metric_iou = BinaryJaccardIndex().to(device)

        # Ensure mask and image are on the same device
        pred_mask_tensor = torch.from_numpy(pred_mask).to(device)
        binary_annotation_tensor = torch.from_numpy(binary_annotation).to(device)

        precision = matric_pre(pred_mask_tensor, binary_annotation_tensor)
        recall = metric_rec(pred_mask_tensor, binary_annotation_tensor)
        accuracy = metric_acc(pred_mask_tensor, binary_annotation_tensor)
        # iou = calculate_iou(binary_annotation, pred_mask)
        dice = calculate_dice(binary_annotation, pred_mask)
        # dice = metric_dic(pred_mask_tensor, binary_annotation_tensor)
        iou = metric_iou(pred_mask_tensor, binary_annotation_tensor)

        precision_scores.append(precision.cpu().numpy())
        recall_scores.append(recall.cpu().numpy())
        accuracy_scores.append(accuracy.cpu().numpy())
        # dice_scores.append(dice.cpu().numpy())
        iou_scores.append(iou.cpu().numpy())
        # iou_scores.append(iou)
        dice_scores.append(dice)

        # Create overlays
        gt_overlay = cv2.addWeighted(
            image, 0.7,
            cv2.merge([binary_annotation * 255, np.zeros_like(binary_annotation), np.zeros_like(binary_annotation)]),
            0.3, 0
        )
        pred_overlay = cv2.addWeighted(
            image, 0.7,
            cv2.merge([np.zeros_like(pred_mask), np.zeros_like(pred_mask), pred_mask * 255]),
            0.3, 0
        )

        # Add labels
        gt_overlay = add_label(gt_overlay, 'Ground Truth', (10, 30))
        pred_overlay = add_label(pred_overlay, 'Prediction', (10, 30))

        # Concatenate images horizontally
        combined_image = np.concatenate((image, gt_overlay, pred_overlay), axis=1)

        # Save the combined image
        save_path = os.path.join(output_dir, f"sample_{i+1}_gt_vs_pred_{chamber}.png")
        cv2.imwrite(save_path, combined_image)

    # Calculate and save mean metrics
    if precision_scores and recall_scores and accuracy_scores and iou_scores and dice_scores:
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_accuracy = np.mean(accuracy_scores)
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        metrics_path = os.path.join(output_dir, "ft_whole_image_public_LA_evaluation_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Mean Precision: {mean_precision:.4f}\n")
            f.write(f"Mean Recall: {mean_recall:.4f}\n")
            f.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
            f.write(f"Mean Dice: {mean_dice:.4f}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n")


def main():
    """Main function to execute the evaluation pipeline."""
    # Load configuration
    config = load_config("/home/hunter.ma/SAM2-ChannelBased_public/fine-tune-train_segment_anything_2_in_60_lines_of_code-main/config/test_configs_whole_image.yaml")

    # Extract configuration parameters
    test_results_dir = config.get("test_results_dir", "test-results")
    chamber = config.get("chamber", "RA")
    csv_file_dir = config.get("csv_file_dir", "data/RA_test.csv").format(chamber)
    test_data_dir = config.get("test_data_dir", "/mnt/rcl-server/workspace/baraa/echo-segmentation-2-resized/")
    sam2_checkpoint = config.get("sam2_checkpoint", "")
    model_cfg = config.get("model_cfg", "sam2_hiera_s.yaml")
    lora_checkpoint = config.get("lora_checkpoint", "")
    use_whole_image = config.get("use_whole_image", True)
    florence_ckpt = config.get("florence_ckpt", "").format(chamber)
    sample_size = config.get("sample_size", 10)
    random_state = config.get("random_state", 42)

    # Validate required configurations
    if not sam2_checkpoint:
        raise ValueError("SAM2 checkpoint path must be provided in config.yaml.")
    if not model_cfg:
        raise ValueError("Model configuration file path must be provided in config.yaml.")
    if not lora_checkpoint:
        raise ValueError("LoRA checkpoint path must be provided in config.yaml.")

    # Create output directory
    create_output_directory(test_results_dir)

    # Load Florence-2 model if needed
    if not use_whole_image:
        if not florence_ckpt:
            raise ValueError("Florence-2 checkpoint path must be provided in config.yaml when use_whole_image is False.")
        florence_model, florence_processor = load_florence2_model(florence_ckpt)
    else:
        florence_model = None
        florence_processor = None

    # Load and sample test data
    sampled_data = load_test_data(csv_file_dir, test_data_dir, sample_size=sample_size, random_state=random_state)

    # Configure the SAM2 model
    sam2_model = configure_sam2_model(model_cfg, sam2_checkpoint, lora_checkpoint)

    # Instantiate the predictor
    predictor = SAM2ImagePredictor(sam2_model)

    # Evaluate the model
    evaluate_model(
        sampled_data=sampled_data,
        predictor=predictor,
        chamber=chamber,
        output_dir=test_results_dir,
        use_whole_image=use_whole_image,
        florence_model=florence_model,
        florence_processor=florence_processor,
        task_prompt='<OD>'
    )

    print("Evaluation completed. Results saved to:", test_results_dir)


if __name__ == "__main__":
    main()
