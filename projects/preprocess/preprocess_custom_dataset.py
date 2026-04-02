"""
Preprocess ImageNet-like datasets into NiT training format.

This script takes a folder containing one or more san-v2 style zip files and
processes them all into the NiT latent format.

Input: a folder with zip files, each zip containing:
  - dataset.json with {"labels": [["00000/img00000000.png", class_idx], ...]}
  - image files organized in subfolders: 00000/img00000000.png, ...

The script auto-discovers zip files and determines their resolution from the
images inside. The highest-resolution zip is used for native-resolution encoding.
Lower-resolution zips (if present) are used for fixed-resolution variants.

Output (NiT training format):
  - Latent .safetensors files per image (with original + h-flipped variants)
  - data_meta JSONL with per-image metadata
  - sampler_meta JSON with packing indices
  - Extracted raw images (for RADIO encoder during training)
  - Ready-to-use training YAML config

Usage:
  python projects/preprocess/preprocess_custom_dataset.py \
      --input-dir datasets \
      --dataset-name imagenet_9to4
"""

import os
import sys
import io
import json
import math
import argparse
import zipfile
import functools
import glob
import numpy as np
import torch
import torch.nn.functional as F

# Ensure nit package is importable even without pip install -e .
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import hflip
from safetensors.torch import save_file
from diffusers import AutoencoderDC


def native_resolution_resize(pil_image, min_image_size, max_image_size):
    """Resize preserving aspect ratio, snapping dims to multiples of min_image_size."""
    w, h = pil_image.size
    if w * h < max_image_size ** 2:
        new_w = max(1, int(w / min_image_size)) * min_image_size
        new_h = max(1, int(h / min_image_size)) * min_image_size
    else:
        new_w = np.sqrt(w / h) * max_image_size
        new_h = new_w * h / w
        new_w = int(new_w / min_image_size) * min_image_size
        new_h = int(new_h / min_image_size) * min_image_size
    pil_image = pil_image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return pil_image


def center_crop_resize(pil_image, image_size):
    """Center-crop to square then resize to image_size."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def dc_ae_encode(dc_ae, images):
    with torch.no_grad():
        latents = dc_ae.encode(images).latent * dc_ae.config.scaling_factor
    return latents


def discover_zips(input_dir):
    """Find all zip files in input_dir and detect their image resolution.

    Returns:
        list of (zip_path, resolution) sorted by resolution descending.
        resolution is (width, height) of images inside.
    """
    zip_paths = sorted(glob.glob(os.path.join(input_dir, "*.zip")))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found in {input_dir}")

    results = []
    for zp in zip_paths:
        zf = zipfile.ZipFile(zp, 'r')
        # Find first image to detect resolution
        img_names = [n for n in zf.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_names:
            print(f"  Skipping {zp}: no images found")
            zf.close()
            continue
        img = Image.open(io.BytesIO(zf.read(img_names[0])))
        res = img.size  # (width, height)
        zf.close()
        results.append((zp, res))
        print(f"  Found: {os.path.basename(zp)} -> {res[0]}x{res[1]}")

    # Sort by resolution (largest first)
    results.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)
    return results


def load_labels_from_zip(zip_path):
    """Load labels from dataset.json inside a zip file.

    Returns:
        list of [image_name, class_label]
    """
    zf = zipfile.ZipFile(zip_path, 'r')
    dataset_json = json.loads(zf.read('dataset.json'))
    labels = dataset_json['labels']
    zf.close()
    return labels


def process_resolution(
    labels, zip_path, dc_ae, device,
    output_latent_dir, resize_fn, data_type,
    vae_downsample_factor=32
):
    """Process all images at a given resolution and save latents + metadata.

    Returns:
        list of metadata dicts for the JSONL file
    """
    os.makedirs(output_latent_dir, exist_ok=True)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    meta_entries = []
    skipped = 0
    encoded = 0
    zf = zipfile.ZipFile(zip_path, 'r')

    for img_name, class_label in tqdm(labels, desc=f"Encoding {data_type}"):
        class_folder = f"class_{class_label:05d}"
        base_name = os.path.splitext(os.path.basename(img_name))[0]
        out_folder = os.path.join(output_latent_dir, class_folder)
        out_path = os.path.join(out_folder, f"{base_name}.safetensors")

        if os.path.exists(out_path):
            # Already encoded — read dimensions from the file to build metadata
            from safetensors.torch import load_file as _load_file
            data = _load_file(out_path)
            latent = data['latent']
            _, _, lh, lw = latent.shape  # [2, C, H, W]
            latent_h = lh
            latent_w = lw
            resized_h = latent_h * vae_downsample_factor
            resized_w = latent_w * vae_downsample_factor
            # Get original dims by opening image (only needed for metadata)
            pil_image = Image.open(io.BytesIO(zf.read(img_name)))
            ori_w, ori_h = pil_image.size
            skipped += 1
        else:
            pil_image = Image.open(io.BytesIO(zf.read(img_name))).convert("RGB")
            ori_w, ori_h = pil_image.size

            # Resize
            pil_image = resize_fn(pil_image)
            resized_w, resized_h = pil_image.size

            # Normalize to tensor [-1, 1]
            img_tensor = normalize(pil_image).unsqueeze(0).to(device, dtype=torch.float32)
            img_hflip = hflip(img_tensor)

            # Encode through VAE
            z_ori = dc_ae_encode(dc_ae, img_tensor)
            z_hflip = dc_ae_encode(dc_ae, img_hflip)
            z = torch.stack([z_ori.squeeze(0), z_hflip.squeeze(0)], dim=0).cpu()

            os.makedirs(out_folder, exist_ok=True)
            save_file(
                {"latent": z.contiguous(), "label": torch.tensor(class_label)},
                out_path
            )

            latent_h = resized_h // vae_downsample_factor
            latent_w = resized_w // vae_downsample_factor
            encoded += 1

        latent_rel = f"{class_folder}/{base_name}.safetensors"
        image_rel = img_name

        meta_entries.append({
            "image_file": image_rel,
            "latent_file": latent_rel,
            "ori_w": ori_w,
            "ori_h": ori_h,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "image_h": resized_h,
            "image_w": resized_w,
            "type": data_type,
        })

    zf.close()
    if skipped > 0:
        print(f"  Skipped {skipped} already-encoded files, encoded {encoded} new files.")
    return meta_entries


def write_jsonl(meta_entries, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as fp:
        for entry in meta_entries:
            fp.write(json.dumps(entry) + '\n')
    print(f"Wrote {len(meta_entries)} entries to {output_path}")


def create_packing(data_meta_path, max_seq_len, output_dir, algorithm='LPFHP'):
    """Generate packing (sampler-meta) JSON from data-meta JSONL."""
    from nit.data.pack import pack_dataset

    with open(data_meta_path, 'r') as fp:
        dataset = [json.loads(line) for line in fp]

    seq_lens = []
    seq_idxs = []
    for idx, data in enumerate(dataset):
        seq_len = int(data['latent_h'] * data['latent_w'])
        seq_lens.append(seq_len)
        seq_idxs.append(idx)

    max_seq_per_pack = max_seq_len
    packed_indices = pack_dataset(algorithm, max_seq_len, max_seq_per_pack, seq_lens, seq_idxs)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(data_meta_path))[0].replace('_meta', '')
    out_path = os.path.join(output_dir, f"{base_name}_{algorithm}_{max_seq_len}.json")
    with open(out_path, 'w') as fp:
        json.dump(packed_indices, fp, indent=4)
    print(f"Wrote packing to {out_path} ({len(packed_indices)} packs)")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ImageNet-like dataset folder into NiT training format."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Folder containing zip files with ImageNet-like datasets. "
             "Each zip should have a dataset.json and image subfolders. "
             "All zips are auto-discovered; highest resolution is used for "
             "native-resolution encoding, others for fixed-resolution variants.",
    )
    parser.add_argument(
        "--dataset-name", type=str, default=None,
        help="Name for the output dataset (used in output paths). "
             "Defaults to the input folder name.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Root output directory. Defaults to same as --input-dir.",
    )
    parser.add_argument(
        "--vae-model", type=str, default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        help="HuggingFace model ID for the DC-AE VAE.",
    )
    parser.add_argument(
        "--min-image-size", type=int, default=32,
        help="Grid size for native-resolution snapping. Default: 32",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=2048,
        help="Max pixel dimension for native-resolution. Default: 2048",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=16384,
        help="Max packed sequence length (in spatial patches). Default: 16384",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for VAE encoding. Default: cuda",
    )
    parser.add_argument(
        "--radio-checkpoint", type=str, default="checkpoints/radio_v2.5-h.pth.tar",
        help="Path to RADIO encoder checkpoint for precomputing features.",
    )
    parser.add_argument(
        "--skip-packing", action="store_true",
        help="Skip the packing step (only encode latents and generate metadata).",
    )
    parser.add_argument(
        "--skip-radio", action="store_true",
        help="Skip RADIO feature precomputation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    dataset_name = args.dataset_name or os.path.basename(os.path.abspath(input_dir))
    output_dir = args.output_dir or input_dir
    dataset_root = os.path.join(output_dir, f"{dataset_name}_nit")
    vae_short_name = args.vae_model.split("/")[-1]

    # Discover zip files
    print(f"Scanning {input_dir} for zip files...")
    zip_list = discover_zips(input_dir)
    print(f"Found {len(zip_list)} zip file(s)\n")

    # Load labels from the first (highest-res) zip
    labels = load_labels_from_zip(zip_list[0][0])
    num_classes = len(set(l[1] for l in labels))
    print(f"Dataset: {len(labels)} images, {num_classes} classes, "
          f"label range: {min(l[1] for l in labels)}-{max(l[1] for l in labels)}")

    all_meta_entries = []
    data_types_used = []
    latent_dirs_used = []
    meta_dir = os.path.join(dataset_root, "data_meta")

    # The highest-resolution zip is used for native-resolution encoding.
    # Any additional zips are treated as fixed-resolution variants.
    native_zip_path, native_res = zip_list[0]

    # Build list of all resolutions to process
    resolutions_to_process = [
        (native_zip_path, native_res, "native-resolution",
         os.path.join(dataset_root, f"{vae_short_name}-native-resolution"),
         functools.partial(native_resolution_resize,
                           min_image_size=args.min_image_size,
                           max_image_size=args.max_image_size)),
    ]
    for zip_path, (w, h) in zip_list[1:]:
        if w != h:
            print(f"Skipping {os.path.basename(zip_path)}: non-square ({w}x{h})")
            continue
        res = w
        resolutions_to_process.append((
            zip_path, (w, h), f"fixed-{res}x{res}",
            os.path.join(dataset_root, f"{vae_short_name}-{res}x{res}"),
            functools.partial(center_crop_resize, image_size=res),
        ))

    # Check if any encoding work is needed (to avoid loading VAE unnecessarily)
    encoding_needed = False
    for zp, res, data_type, latent_dir, resize_fn in resolutions_to_process:
        sample_label = labels[0]
        class_folder = f"class_{sample_label[1]:05d}"
        base_name = os.path.splitext(os.path.basename(sample_label[0]))[0]
        sample_path = os.path.join(latent_dir, class_folder, f"{base_name}.safetensors")
        if not os.path.exists(sample_path):
            encoding_needed = True
            break

    # Load VAE only if encoding is needed
    if encoding_needed:
        print(f"\nLoading VAE: {args.vae_model}")
        dc_ae = AutoencoderDC.from_pretrained(args.vae_model).to(args.device, dtype=torch.float32)
        dc_ae.eval()
        dc_ae.requires_grad_(False)
    else:
        print("\nAll latent files already exist, skipping VAE loading.")
        dc_ae = None

    # --- Process each resolution ---
    for zp, res, data_type, latent_dir, resize_fn in resolutions_to_process:
        meta_suffix = "nr" if data_type == "native-resolution" else f"{res[0]}x{res[1]}"
        meta_path = os.path.join(meta_dir, f"{vae_short_name}_{meta_suffix}_meta.jsonl")

        print(f"\n=== Processing {data_type} (from {res[0]}x{res[1]} source) ===")
        meta = process_resolution(
            labels, zp, dc_ae, args.device,
            latent_dir, resize_fn, data_type,
        )
        all_meta_entries.extend(meta)
        data_types_used.append(data_type)
        latent_dirs_used.append(latent_dir)
        write_jsonl(meta, meta_path)

    # --- Merged meta ---
    merged_meta_path = os.path.join(meta_dir, f"{vae_short_name}_merge_meta.jsonl")
    write_jsonl(all_meta_entries, merged_meta_path)

    # --- Extract images for RADIO encoder (training needs raw images) ---
    print("\n=== Extracting raw images for RADIO encoder ===")
    image_dir = os.path.join(dataset_root, "images")
    zf_extract = zipfile.ZipFile(native_zip_path, 'r')
    for img_name, _ in tqdm(labels, desc="Extracting images"):
        out_path = os.path.join(image_dir, img_name)
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf_extract.open(img_name) as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    zf_extract.close()
    print(f"Extracted images to {image_dir}")

    # --- Precompute RADIO features ---
    radio_feature_dir = os.path.join(dataset_root, "radio-features")
    if not args.skip_radio:
        print(f"\n=== Precomputing RADIO features ===")
        from nit.models.nvidia_radio.hubconf import radio_model
        radio_dtype = torch.bfloat16
        encoder = radio_model(
            version=args.radio_checkpoint, progress=True, support_packing=True
        )
        encoder.to(device=args.device, dtype=radio_dtype).eval()
        encoder.requires_grad_(False)

        to_tensor = transforms.ToTensor()
        radio_skipped = 0
        radio_encoded = 0
        for entry in tqdm(all_meta_entries, desc="RADIO features"):
            output_file = os.path.join(radio_feature_dir, entry['latent_file'])
            if os.path.exists(output_file):
                radio_skipped += 1
                continue

            image_file = os.path.join(image_dir, entry['image_file'])
            height = entry['latent_h'] * 16
            width = entry['latent_w'] * 16
            data_type = entry['type']

            if data_type == 'native-resolution':
                preprocess = functools.partial(native_resolution_resize,
                                               min_image_size=args.min_image_size,
                                               max_image_size=args.max_image_size)
            else:
                assert height == width
                preprocess = functools.partial(center_crop_resize, image_size=height)

            pil_image = preprocess(pil_image=Image.open(image_file).convert("RGB"))
            pil_flipped = hflip(pil_image)

            img_ori = to_tensor(pil_image).unsqueeze(0).to(device=args.device, dtype=radio_dtype)
            img_flip = to_tensor(pil_flipped).unsqueeze(0).to(device=args.device, dtype=radio_dtype)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=radio_dtype):
                _, feat_ori = encoder.forward_pack([img_ori])
                _, feat_flip = encoder.forward_pack([img_flip])

            radio_feature = torch.stack([feat_ori.cpu(), feat_flip.cpu()], dim=0).to(torch.float32)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            save_file({'radio_feature': radio_feature}, output_file)
            radio_encoded += 1

        del encoder
        torch.cuda.empty_cache()
        print(f"  RADIO: encoded {radio_encoded}, skipped {radio_skipped} existing.")

    # --- Packing ---
    if not args.skip_packing:
        sampler_dir = os.path.join(dataset_root, "sampler_meta")
        # Generate packing for the requested max_seq_len
        print(f"\n=== Generating packing (max_seq_len={args.max_seq_len}) ===")
        packed_json_path = create_packing(
            merged_meta_path, args.max_seq_len, sampler_dir
        )
        # Also generate 8192 packing (for NiT-XXL) if not already covered
        if args.max_seq_len != 8192:
            print(f"\n=== Generating packing (max_seq_len=8192, for NiT-XXL) ===")
            create_packing(merged_meta_path, 8192, sampler_dir)
    else:
        packed_json_path = "<run pack_dataset.py separately>"

    # --- Generate training config ---
    config = {
        "model": {
            "transport": {"path_type": "linear", "prediction": "v", "weighting": "lognormal"},
            "network": {
                "target": "nit.models.c2i.nit_model.NiT",
                "params": {
                    "class_dropout_prob": 0.1,
                    "num_classes": num_classes,
                    "depth": 28,
                    "hidden_size": 1152,
                    "patch_size": 1,
                    "in_channels": 32,
                    "num_heads": 16,
                    "qk_norm": True,
                    "encoder_depth": 8,
                    "z_dim": 1280,
                    "use_checkpoint": False,
                },
            },
            "vae_dir": args.vae_model,
            "slice_vae": False,
            "tile_vae": False,
            "enc_type": "radio",
            "enc_dir": "checkpoints/radio_v2.5-h.pth.tar",
            "proj_coeff": 1.0,
            "use_ema": True,
            "ema_decay": 0.9999,
        },
        "data": {
            "data_type": "improved_pack",
            "dataset": {
                "packed_json": packed_json_path,
                "jsonl_dir": merged_meta_path,
                "data_types": data_types_used,
                "latent_dirs": latent_dirs_used,
                "image_dir": image_dir,
                "radio_feature_dir": radio_feature_dir,
            },
            "dataloader": {"num_workers": 4, "batch_size": 1},
        },
        "training": {
            "tracker": None,
            "tracker_kwargs": {"wandb": {"group": "c2i"}},
            "max_train_steps": 2000000,
            "checkpointing_steps": 2000,
            "checkpoints_total_limit": 2,
            "resume_from_checkpoint": "latest",
            "learning_rate": 5.0e-5,
            "learning_rate_base_batch_size": 4,
            "scale_lr": True,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "target": "torch.optim.AdamW",
                "params": {"betas": [0.9, 0.95], "weight_decay": 1.0e-2, "eps": 1.0e-6},
            },
            "max_grad_norm": 1.0,
            "proportion_empty_prompts": 0.0,
            "mixed_precision": "bf16",
            "allow_tf32": True,
            "validation_steps": 500,
            "checkpoint_list": [200000, 500000, 1000000, 1500000],
        },
    }

    config_path = os.path.join(dataset_root, f"train_config.yaml")
    try:
        from omegaconf import OmegaConf
        OmegaConf.save(config=OmegaConf.create(config), f=config_path)
    except ImportError:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    print(f"\nGenerated training config: {config_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Input folder:     {input_dir}")
    print(f"Dataset name:     {dataset_name}")
    print(f"Output root:      {dataset_root}")
    print(f"Images:           {len(labels)}")
    print(f"Classes:          {num_classes}")
    print(f"Zip files used:   {len(zip_list)}")
    print(f"Resolutions:      {', '.join(data_types_used)}")
    print(f"Latent dirs:      {latent_dirs_used}")
    print(f"Merged meta:      {merged_meta_path}")
    print(f"Packing JSON:     {packed_json_path}")
    print(f"RADIO features:   {radio_feature_dir}")
    print(f"Raw images:       {image_dir}")
    print(f"Training config:  {config_path}")
    print(f"\nTo train:")
    print(f"  accelerate launch projects/train/packed_trainer_c2i.py \\")
    print(f"    --config {config_path} \\")
    print(f"    --project_dir workdir/{dataset_name}")


if __name__ == "__main__":
    main()
