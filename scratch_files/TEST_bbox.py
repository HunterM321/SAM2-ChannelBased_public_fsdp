import numpy as np
import pandas as pd
import torch
import cv2
import os
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Create a directory for saving test results
os.makedirs("test-results", exist_ok=True)

chamber = "RA"
# Load the Test Data
test_csv_file = r"data/{}_test.csv".format(chamber)  # Replace with your test CSV file path
test_data_dir = r"/mnt/rcl-server/workspace/baraa/echo-segmentation-2-resized/"  # Path to test dataset

# Read the test CSV file
test_df = pd.read_csv(test_csv_file)

# Sample a subset of the test data
sample_size = 10  # You can change this to the number of samples you want
sampled_test_df = test_df.sample(n=sample_size, random_state=42)  # random_state for reproducibility

# List to store sampled image and annotation paths
sampled_data = []

# Iterate through the filenames in the sampled dataframe
for index, row in sampled_test_df.iterrows():
    filename = row['file_name']  # Access the filename from the 'file_name' column in the CSV
    image_path = os.path.join(test_data_dir, "images/", filename + ".png")
    annotation_path = os.path.join(test_data_dir, "masks/", filename + ".png")
    
    # Add to the data list
    sampled_data.append({"image": image_path, "annotation": annotation_path})

# Load the model
sam2_checkpoint = "/home/baraa.abdelsamad/SAM2-ChannelBased/ckpts/sam2_hiera_small.pt"  # Path to the trained model checkpoint
model_cfg = "sam2_hiera_s.yaml"  # Path to the model configuration file


# sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # Load model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # Load model
# Define the LoRA configuration (should match the one used during training)
config = LoraConfig(
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
        "memory_attention.layers.0.cross_attn_image.out_proj"
    ],
    lora_dropout=0.0,
    bias="none",
)

# Apply the LoRA configuration to the model
model = get_peft_model(sam2_model, config)

lora_checkpoint = torch.load('/home/baraa.abdelsamad/SAM2-ChannelBased/model.torch')
# Extract all keys from the checkpoint that start with 'base_model.model'
# state_dict = {k: v for k, v in lora_checkpoint.items() if k.startswith('base_model.model')}

# # Optionally, if you need to remove the 'base_model.model' prefix to match your model's state dict:
# new_state_dict = {k[len('base_model.model.'):]: v for k, v in state_dict.items()}

model.load_state_dict(lora_checkpoint)
# Instantiate the predictor
predictor = SAM2ImagePredictor(model)


# Define a function to calculate IoU
def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

# Define a function to calculate Dice coefficient
def calculate_dice(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    dice = (2.0 * intersection) / (gt_mask.sum() + pred_mask.sum())
    return dice

# Define a function to extract bounding box from the mask
def get_bbox(mask):
    # mask_channel = mask[..., 0]  # Use the first channel of the mask do this for LA 
    
    # Get all coordinates in the mask where the value is greater than 0
    coords = np.argwhere(mask > 0)
    
    if coords.size == 0:
        return None  # Return None if the mask is empty or has no positive values

    # Calculate bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return [x_min, y_min, x_max, y_max]


def add_label(image, label, position, font_scale=0.8, thickness=2, color=(255, 255, 255)):
    """
    Add a label to the image at the specified position.
    
    Args:
    - image (numpy array): The image to add the label to.
    - label (str): The text label.
    - position (tuple): The (x, y) position for the label.
    - font_scale (float): Font scale (size) of the label text.
    - thickness (int): Thickness of the label text.
    - color (tuple): Color of the text in BGR format.
    
    Returns:
    - image (numpy array): The image with the label added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

def binarize_mask_by_highest_label(mask):
    """
    Binarize the mask by converting the highest label (in terms of pixel intensity) to 1 and others to 0.
    
    Args:
    - mask (numpy array): The input mask.
    
    Returns:
    - binary_mask (numpy array): The binarized mask with labels 0 and 1.
    """
    highest_label = np.max(mask)  # Find the highest label in the mask
    
    # Binarize the mask: Convert the highest label to 1, others to 0
    binary_mask = np.where(mask == highest_label, 1, 0).astype(np.uint8)
    
    return binary_mask

# Evaluate on sampled data
iou_scores = []
dice_scores = []
predictor.model.eval()  # Set model to evaluation mode

for i, sample in enumerate(sampled_data):
    # Load image and annotation
    image = cv2.imread(sample["image"], 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotation = cv2.imread(sample["annotation"], 1)  # Load mask as grayscale
    print(f"is there a mask: {np.unique(annotation[...,2])}")
    # annotation = annotation[...,0] # for LV annotation = cv2.imread(sample["annotation"], 1) 
    annotation = annotation[...,2] # for RA 
    print(f"mask shape {annotation.shape}")

    # Get bounding box from the annotation mask
    bbox = get_bbox(annotation)
    print(f"got bbox: {bbox}")

    annotation = binarize_mask_by_highest_label(annotation)
    print(f"mask after binarizing: {np.unique(annotation)}")

    if bbox is None:
        continue  # Skip if no valid bounding box is found

    predictor.set_image(image_rgb)
    prd_masks, scores, logits = predictor.predict(
        box=bbox
    )
    print(f"shape of pred mask: {prd_masks.shape}")
    # Sort predicted masks by score
    # Sort the predicted masks by their scores
    sorted_masks = prd_masks[np.argsort(scores)][::-1].astype(bool)

    # Assuming the highest scoring mask is the most accurate
    pred_mask = sorted_masks[0].astype(np.uint8)
    print(f"pred mask unique is: {np.unique(pred_mask)}")

    # Compare ground truth mask with predicted mask
    iou = calculate_iou(annotation, pred_mask)
    dice = calculate_dice(annotation, pred_mask)
    iou_scores.append(iou)
    dice_scores.append(dice)

    # Overlay masks with specific colors (e.g., blue for GT, red for prediction)
    gt_overlay = cv2.addWeighted(image, 0.7, cv2.merge([annotation * 255, np.zeros_like(annotation), np.zeros_like(annotation)]), 0.3, 0)
    pred_overlay = cv2.addWeighted(image, 0.7, cv2.merge([np.zeros_like(pred_mask), np.zeros_like(pred_mask), pred_mask * 255]), 0.3, 0)

    # Add labels to the overlays
    gt_overlay = add_label(gt_overlay, 'Ground Truth', (10, 30))
    pred_overlay = add_label(pred_overlay, 'Prediction', (10, 30))

    # Concatenate original, ground truth overlay, and predicted overlay
    combined_image = np.concatenate((image, gt_overlay, pred_overlay), axis=1)

    # Save the combined image
    save_path = os.path.join("test-results", f"sample_{i+1}_gt_vs_pred_{chamber}.png")
    cv2.imwrite(save_path, combined_image)

# Calculate mean IoU and mean Dice score
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)

print(f"Mean IoU: {mean_iou}")
print(f"Mean Dice: {mean_dice}")
