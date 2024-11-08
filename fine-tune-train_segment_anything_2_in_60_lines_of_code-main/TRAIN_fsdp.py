# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import os
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy

from peft import LoraConfig, get_peft_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Read data
# DO LATER
# data_dir=r"LabPicsV1//" # Path to dataset (LabPics 1)
# data=[] # list of files in dataset
# for ff, name in enumerate(os.listdir(data_dir+"Simple/Train/Image/")):  # go over all folder annotation
#     data.append({"image":data_dir+"Simple/Train/Image/"+name,"annotation":data_dir+"Simple/Train/Instance/"+name[:-4]+".png"})

def setup_fsdp(rank, world_size):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_fsdp():
    """Destroy the process group after training."""
    dist.destroy_process_group()

def print_model_summary(model):
        # Total and trainable parameter counters
        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
                trainable_status = "Yes"
            else:
                trainable_status = "No"

        print("=" * 100)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Calculate and print the percentage of trainable parameters
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"Trainable Percentage: {trainable_percentage:.2f}%")
        print(f"Frozen Percentage: {100 - trainable_percentage:.2f}%")

def read_batch(data): # read random image and its annotation from the dataset (LabPics)

    # select image
    ent = data[np.random.randint(len(data))] # choose random entry
    Img = cv2.imread(ent["image"],1)  # read image
    # Convert BGR to RGB
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # print(Img.shape)

    ann_map = cv2.imread(ent["annotation"],1) # read annotation

    # resize image
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scaling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    # print(f"img shape = {Img.shape}")
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    # print(f"mask shape: {ann_map.shape}")

    # merge vessels and materials annotations
    mat_map = ann_map[:,:,0] # material annotation map
    # print(mat_map.shape)
    ves_map = ann_map[:,:,2] # vessel annotation map
    # print(ves_map.shape)
    mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps
    # print(np.unique(ann_map))

    # Get binary masks, points, and bounding boxes
    inds = np.unique(mat_map)[1:] # load all indices
    points = []
    masks = []
    bboxes = []
    
    img_height, img_width = Img.shape[:2]  # get the dimensions of the image

    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
        points.append([[yx[1], yx[0]]])

        # Set bounding box to the entire image boundary
        use_bbox = False

        if use_bbox:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = [x_min, y_min, x_max, y_max]
        else:
            bbox = [0, 0, img_width, img_height]
        
        bboxes.append(bbox)
    
    return Img, np.array(masks), np.array(points), np.array(bboxes)

def should_wrap(module, recurse, nonwrapped_numel):
    # Only wrap trainable layers; skip parts relying on (H, W, C) input.
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        return True  # Wrap these
    if isinstance(module, nn.ModuleList) or isinstance(module, SAM2ImagePredictor):
        return False  # Avoid wrapping these
    return any(p.requires_grad for p in module.parameters())

def train(rank, world_size, data, num_epochs=50):
    setup_fsdp(rank, world_size)
    
    sam2_checkpoint = "/home/hunter.ma/SAM2-ChannelBased_public_fsdp/important_files/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=f"cuda:{rank}")

    # Define the LoRA configuration
    config = LoraConfig(
        r=8, 
        lora_alpha=8,   
        target_modules=[
            # sam_mask_decoder layers
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
            
            # memory_attention layers
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
    print(model.device)

    lora_checkpoint = torch.load(
        '/home/hunter.ma/SAM2-ChannelBased_public_fsdp/prev_results/test-results-whole-image/model.torch',
        # map_location=f"cuda:{rank}"
    )
    print("Loaded model")

    print(model.image_encoder.trunk.patch_embed.proj.weight.shape)
    model.load_state_dict(lora_checkpoint)
    print(model.image_encoder.trunk.patch_embed.proj.weight.shape)
    # print(f"model summary --:-- {model.summary()}")

    # Wrap the model with FSDP
    # fsdp_policy = always_wrap_policy(module=model)
    model = FSDP(
        model,
        device_id=rank,
        use_orig_params=True
    )
    print(model.image_encoder.trunk.patch_embed.proj.weight.shape)

    # Call the function to print summary and parameter stats
    print_model_summary(model)

    predictor = SAM2ImagePredictor(model)
    print(predictor)
    # exit()

    # do lora here

    # Set training parameters

    # predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
    # predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
    optimizer = optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler()



    # Training loop

    num_epochs = 50
    num_iterations_per_epoch = len(data) // world_size

    for epoch in range(num_epochs):
        epoch_loss = 0
        mean_iou = 0
        
        # Create a progress bar for the current epoch
        with tqdm(total=num_iterations_per_epoch, desc=f"Rank {rank} Epoch {epoch+1}/{num_epochs}") as pbar:
            for itr in range(num_iterations_per_epoch):
                with torch.cuda.amp.autocast():  # cast to mixed precision
                    image, mask, input_point, bboxes = read_batch(data)  # load data batch
                    if mask.shape[0] == 0: 
                        continue  # ignore empty batches
                    
                    predictor.set_image(image)  # apply SAM image encoder to the image

                    # prompt encoding with bounding boxes instead of points
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                        point_coords=None, point_labels=None, box=bboxes, mask_logits=None, normalize_coords=True
                    )  # now using bbox instead of point prompts

                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=None, boxes=unnorm_box, masks=None,
                    )  # pass bboxes instead of points

                    # mask decoder
                    batched_mode = unnorm_box.shape[0] > 1  # multi-object prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True, repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image resolution

                    # Segmentation Loss calculation
                    gt_mask = torch.tensor(mask.astype(np.float32)).to(f"cuda:{rank}")
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)).mean()  # cross-entropy loss

                    # Score loss calculation (intersection over union) IOUxe
                    inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
                    union = gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
                    iou = inter / (union + 1e-6)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss * 0.05
                    
                    epoch_loss += loss.item()

                    # apply backpropagation
                    predictor.model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Update progress bar and print status
                    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                    pbar.set_postfix({"Loss": epoch_loss / (itr + 1), "Mean IOU": mean_iou})
                    pbar.update(1)

                if itr % 1000 == 0 and rank == 0:
                    torch.save(predictor.model.state_dict(), "ft_whole_image_model_multi.torch")
                    print("Model saved.")

        print(f"Rank {rank} Epoch {epoch+1}/{num_epochs} completed with Mean IOU: {mean_iou}")
    
    cleanup_fsdp()

def main():
    # Path to the CSV file and dataset directory
    csv_file = r"/home/hunter.ma/SAM2-ChannelBased_public_fsdp/data/train.csv"  # Replace with your CSV file path
    data_dir = r"/home/hunter.ma/CAMUS-processesd/"  # Path to dataset (LabPics 1)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # List to store image and annotation paths
    data = []

    # Iterate through the filenames in the CSV
    for index, row in df.iterrows():
        filename = row['file_name']  # Access the filename from the 'filename' column in the CSV
        image_path = os.path.join(data_dir, "images/", filename)
        annotation_path = os.path.join(data_dir, "masks/", filename)
        
        # Add to the data list
        data.append({"image": image_path, "annotation": annotation_path})

    # Optionally, print the first few entries to verify
    # print(data[:5])

    world_size = torch.cuda.device_count()
    print(world_size)

    mp.spawn(train, args=(world_size, data), nprocs=world_size, join=True)

    print("Training completed.")

if __name__ == "__main__":
    main()
