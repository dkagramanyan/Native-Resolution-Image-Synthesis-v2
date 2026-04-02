#!/usr/bin/env python
"""Precompute RADIO encoder features for all images in the dataset.

Saves per-sample safetensors files with 'radio_feature' key of shape
[2, num_tokens, feature_dim] (index 0 = original, index 1 = horizontally flipped).

Usage:
    python projects/preprocess/precompute_radio_features.py \
        --jsonl datasets/imagenet_9to4_nit/data_meta/dc-ae-f32c32-sana-1.1-diffusers_merge_meta.jsonl \
        --image_dir datasets/imagenet_9to4_nit/images \
        --output_dir datasets/imagenet_9to4_nit/radio-features \
        --radio_checkpoint checkpoints/radio_v2.5-h.pth.tar \
        --batch_size 8
"""

import argparse
import json
import os
from functools import partial

import torch
from PIL import Image
from safetensors.torch import save_file
from torchvision import transforms
from torchvision.transforms.functional import hflip
from tqdm import tqdm


def resize_arr(pil_image, height, width):
    return pil_image.resize((width, height), resample=Image.Resampling.BICUBIC)


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    arr = __import__('numpy').array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to dataset metadata jsonl")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for radio features")
    parser.add_argument("--radio_checkpoint", type=str, default="checkpoints/radio_v2.5-h.pth.tar")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Load dataset metadata
    with open(args.jsonl, 'r') as fp:
        dataset = [json.loads(line) for line in fp]
    print(f"Loaded {len(dataset)} samples from {args.jsonl}")

    # Load RADIO encoder
    from nit.models.nvidia_radio.hubconf import radio_model
    encoder = radio_model(version=args.radio_checkpoint, progress=True, support_packing=True)
    encoder.to(device=device, dtype=dtype).eval()
    encoder.requires_grad_(False)
    print("RADIO encoder loaded")

    to_tensor = transforms.ToTensor()
    skipped = 0

    for sample_idx in tqdm(range(len(dataset)), desc="Precomputing RADIO features"):
        data_meta = dataset[sample_idx]
        # Output path mirrors latent_file structure
        output_file = os.path.join(args.output_dir, data_meta['latent_file'])
        if os.path.exists(output_file):
            skipped += 1
            continue

        # Load and preprocess image (same as training pipeline)
        image_file = os.path.join(args.image_dir, data_meta['image_file'])
        height = data_meta['latent_h'] * 16
        width = data_meta['latent_w'] * 16
        data_type = data_meta['type']

        if data_type == 'native-resolution':
            preprocess = partial(resize_arr, height=height, width=width)
        else:
            assert height == width
            preprocess = partial(center_crop_arr, image_size=height)

        pil_image = preprocess(pil_image=Image.open(image_file).convert("RGB"))
        pil_flipped = hflip(pil_image)

        # Convert to [0, 1] tensors for RADIO (RADIO expects [0, 1] input)
        img_ori = to_tensor(pil_image).unsqueeze(0).to(device=device, dtype=dtype)
        img_flip = to_tensor(pil_flipped).unsqueeze(0).to(device=device, dtype=dtype)

        # Run RADIO encoder
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            _, feat_ori = encoder.forward_pack([img_ori])
            _, feat_flip = encoder.forward_pack([img_flip])

        # Stack: [2, num_tokens, feature_dim]
        radio_feature = torch.stack([feat_ori.cpu(), feat_flip.cpu()], dim=0).to(torch.float32)

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file({'radio_feature': radio_feature}, output_file)

    print(f"Done. Processed {len(dataset) - skipped}, skipped {skipped} existing files.")


if __name__ == "__main__":
    main()
