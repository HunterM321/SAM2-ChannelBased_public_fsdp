import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Function to resize images and masks
def resize_images_and_masks(directory,save_directory, target_size):
    # List all .png files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    # Add a progress bar using tqdm
    for file_name in tqdm(files, desc="Resizing images"):
        file_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(save_directory, file_name)
        with Image.open(file_path) as img:
            # Print original size
            # print("Original size of {}: {}".format(file_name, img.size))
            
            # Resize image
            resized_img = img.resize(target_size, Image.ANTIALIAS)
            resized_img.save(new_file_path)
            # print("Resized {} to {}".format(file_name, target_size))

# Function to assign different channels to chambers in masks
def process_masks_channel_based(directory):
    # List all .png files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.png') and ('RA' in f or 'LA' in f or 'LV' in f)]
    
    # Add a progress bar using tqdm
    for file_name in tqdm(files, desc="Processing masks"):
        file_path = os.path.join(directory, file_name)
        with Image.open(file_path) as img:
            # Convert image to a NumPy array
            mask_array = np.array(img)
            
            # Create an empty 3-channel array (initialize to black)
            processed_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
            
            # Assign channels based on chamber names
            if 'RA' in file_name:
                processed_mask[..., 0] = mask_array  # Red channel for RA
            if 'LA' in file_name:
                processed_mask[..., 1] = mask_array  # Green channel for LA
            if 'LV' in file_name:
                processed_mask[..., 2] = mask_array  # Blue channel for LV
            
            # Convert back to an image
            processed_img = Image.fromarray(processed_mask)
            processed_img.save(file_path)
            # print("Processed {} with multi-channel mask.".format(file_name))

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Function to assign different numbers to chambers in masks
def process_masks_number_based(directory):
    # List all .png files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.png') and ('RA' in f or 'LA' in f or 'LV' in f)]
    
    # Add a progress bar using tqdm
    for file_name in tqdm(files, desc="Processing masks"):
        file_path = os.path.join(directory, file_name)
        with Image.open(file_path) as img:
            # Convert image to a NumPy array
            mask_array = np.array(img)
            
            # Initialize an empty single-channel array (same size as mask_array)
            processed_mask = np.zeros(mask_array.shape, dtype=np.uint8)
            
            # Assign different numbers based on chamber names
            if 'RA' in file_name:
                processed_mask[mask_array > 0] = 1  # Assign 1 for RA
            if 'LA' in file_name:
                processed_mask[mask_array > 0] = 2  # Assign 2 for LA
            if 'LV' in file_name:
                processed_mask[mask_array > 0] = 3  # Assign 3 for LV
            
            # Convert back to an image
            processed_img = Image.fromarray(processed_mask)
            processed_img.save(file_path)

# Example usage
# process_masks("/path/to/your/mask/directory")


# Example usage
if __name__ == "__main__":
    # Directory containing the images and masks
    # images_directory = '/mnt/rcl-server/workspace/baraa/echo-segmentation-2/images/'
    # masks_directory = '/mnt/rcl-server/workspace/baraa/echo-segmentation-2/masks/'
    # images_save_directory = '/mnt/rcl-server/workspace/baraa/echo-segmentation-2-resized/images/'
    masks_save_directory = '/mnt/rcl-server/workspace/baraa/echo-segmentation-2-resized/masks/'
    # Define target size for resizing (width, height)
    target_size = (256, 256)  # Example size, adjust as needed

    # Step 1: Resize images and masks
    # resize_images_and_masks(images_directory,images_save_directory, target_size)
    # resize_images_and_masks(masks_directory,masks_save_directory, target_size)

    # Step 2: Process masks to assign different channels to RA, LA, LV
    process_masks_number_based(masks_save_directory)
# nohup python offline_preprocessing/preprocess.py > output.log 2>&1 &