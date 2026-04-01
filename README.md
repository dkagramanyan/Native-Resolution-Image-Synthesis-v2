# Native-Resolution Image Synthesis (NiT) v2


## Training (step by step)

### Step 1. Install environment

```bash
conda create -n nit_env python=3.10
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn
pip install -r requirements.txt
pip install -e .
```

### Step 2. Check environment and download models

```bash
python check_environment.py
```

This checks all Python packages, verifies GPU availability, and automatically downloads missing models:
- **DC-AE VAE** (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`) — for encoding images to latents
- **RADIO v2.5-h** — image encoder for REPA loss during training

Run with `--no-download` to only check without downloading.

### Step 3. Preprocess dataset

Place your ImageNet-like zip files into `datasets/`. Each zip must contain a `dataset.json` and image subfolders:
```
datasets/
├── imagenet_9to4_1024x1024_1024x1024.zip   # highest resolution -> native-resolution encoding
├── imagenet_9to4_1024x1024_512x512.zip     # optional -> fixed-512x512 variant
└── imagenet_9to4_1024x1024_256x256.zip     # optional -> fixed-256x256 variant
```

Each zip has this structure:
```
my_dataset.zip
├── dataset.json    # {"labels": [["00000/img00000000.png", 0], ...]}
├── 00000/
│   ├── img00000000.png
│   └── ...
└── 00001/
    └── ...
```

Run preprocessing (encodes images through DC-AE VAE, generates metadata, packing, and training config):
```bash
python projects/preprocess/preprocess_custom_dataset.py \
    --input-dir datasets \
    --dataset-name imagenet_9to4
```

Output:
```
datasets/imagenet_9to4_nit/
├── dc-ae-f32c32-sana-1.1-diffusers-native-resolution/
│   ├── class_00000/img00000000.safetensors
│   ├── class_00001/...
│   └── class_00002/...
├── dc-ae-f32c32-sana-1.1-diffusers-512x512/
├── dc-ae-f32c32-sana-1.1-diffusers-256x256/
├── images/                    # extracted raw images (for RADIO encoder)
├── data_meta/
│   └── ..._merge_meta.jsonl
├── sampler_meta/
│   └── ..._merge_LPFHP_16384.json
└── train_config.yaml          # ready-to-use training config
```

Options:
- `--dataset-name NAME` — custom name for output (default: input folder name)
- `--output-dir DIR` — where to write outputs (default: same as `--input-dir`)
- `--max-seq-len N` — max packed sequence length in spatial patches (default: 16384)
- `--skip-packing` — only encode latents, skip packing step

`num_classes` is set automatically from dataset labels. Adjust if fine-tuning from an ImageNet-pretrained checkpoint.

### Step 4. Train

```bash
torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:60563 \
    projects/train/packed_trainer_c2i.py \
    --config datasets/imagenet_9to4_nit/train_config.yaml \
    --project_dir workdir/imagenet_9to4 \
    --seed 0
```

Adjust `--nproc_per_node` to match your number of GPUs. Checkpoints are saved to `workdir/imagenet_9to4/checkpoints/`.


## Sampling (step by step)

### Step 1. Generate samples

Use your trained checkpoint from `workdir/imagenet_9to4/checkpoints/`. Adjust `--nproc_per_node`, `--height`, `--width`, and `--config` as needed.

256x256 (SDE sampler):
```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  projects/sample/sample_c2i_ddp.py \
  --config datasets/imagenet_9to4_nit/train_config.yaml \
  --ckpt workdir/imagenet_9to4/checkpoints/checkpoint-<STEP>/model.safetensors \
  --sample-dir ./samples \
  --height 256 \
  --width 256 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 2.25 \
  --guidance-low 0.0 \
  --guidance-high 0.7 \
  --slice_vae
```

512x512 (SDE sampler):
```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  projects/sample/sample_c2i_ddp.py \
  --config datasets/imagenet_9to4_nit/train_config.yaml \
  --ckpt workdir/imagenet_9to4/checkpoints/checkpoint-<STEP>/model.safetensors \
  --sample-dir ./samples \
  --height 512 \
  --width 512 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 2.05 \
  --guidance-low 0.0 \
  --guidance-high 0.7 \
  --slice_vae
```

768x768 (ODE sampler):
```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  projects/sample/sample_c2i_ddp.py \
  --config datasets/imagenet_9to4_nit/train_config.yaml \
  --ckpt workdir/imagenet_9to4/checkpoints/checkpoint-<STEP>/model.safetensors \
  --sample-dir ./samples \
  --height 768 \
  --width 768 \
  --per-proc-batch-size 32 \
  --mode ode \
  --num-steps 50 \
  --cfg-scale 3.0 \
  --guidance-low 0.0 \
  --guidance-high 0.7 \
  --slice_vae
```

Replace `<STEP>` with your checkpoint step number (e.g. `checkpoint-200000`).

### Reference: sampling hyper-parameters

| Resolution | Solver | NFE | CFG-scale | CFG-interval | FID | IS |
|------------|--------|-----|-----------|--------------|-----|----|
| 256x256 | SDE | 250 | 2.25 | [0.0, 0.7] | 2.16 | 253.44 |
| 512x512 | SDE | 250 | 2.05 | [0.0, 0.7] | 1.57 | 260.69 |
| 768x768 | ODE | 50 | 3.0 | [0.0, 0.7] | 4.05 | 262.31 |
| 1024x1024 | ODE | 50 | 3.0 | [0.0, 0.8] | 4.52 | 286.87 |
| 432x768 | ODE | 50 | 2.75 | [0.0, 0.7] | 4.11 | 254.71 |
| 480x640 | ODE | 50 | 2.75 | [0.0, 0.7] | 3.72 | 284.94 |
| 640x480 | ODE | 50 | 2.5 | [0.0, 0.7] | 3.41 | 259.06 |
