# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

# updated to fit lora

def _load_checkpoint(model, ckpt_path):
    # print(f"model is : {model}")
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


# def _load_checkpoint(model, ckpt_path):
#     if ckpt_path is not None:
#         # Load the entire checkpoint
#         checkpoint = torch.load(ckpt_path, map_location="cpu")

#         # Initialize a new state dictionary
#         new_state_dict = {}

#         # Strip the 'base_model.model.' prefix and the '.base_layer' suffix
#         for key, value in checkpoint.items():
#             # Remove the prefix
#             if key.startswith("base_model.model."):
#                 new_key = key[len("base_model.model."):]
#             else:
#                 new_key = key

#             # Remove the suffix
#             if new_key.endswith(".base_layer.weight"):
#                 new_key = new_key.replace(".base_layer.weight", ".weight")
#             if new_key.endswith(".base_layer.bias"):
#                 new_key = new_key.replace(".base_layer.bias", ".bias")

#             # Add the new key-value pair to the new state dictionary
#             new_state_dict[new_key] = value

#         # Load the modified state dictionary into the model
#         missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

#         # Handle missing or unexpected keys
#         if missing_keys:
#             logging.error(f"Missing keys: {missing_keys}")
#             raise RuntimeError(f"Failed to load checkpoint: missing keys: {missing_keys}")
#         if unexpected_keys:
#             logging.error(f"Unexpected keys: {unexpected_keys}")
#             raise RuntimeError(f"Failed to load checkpoint: unexpected keys: {unexpected_keys}")

#         logging.info("Loaded checkpoint successfully")
