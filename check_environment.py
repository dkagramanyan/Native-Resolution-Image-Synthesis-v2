"""
Check that all dependencies are installed and download required models.

Usage:
    python check_environment.py              # check everything
    python check_environment.py --download   # also download models
"""

import sys
import argparse
import importlib
import os
import subprocess


# ── Required Python packages ──────────────────────────────────────────────────
# (module_to_import, pip_package_name, required_for)
REQUIRED_PACKAGES = [
    ("torch", "torch", "core"),
    ("torchvision", "torchvision", "core"),
    ("accelerate", "accelerate", "training"),
    ("diffusers", "diffusers", "preprocessing + training"),
    ("transformers", "transformers", "training"),
    ("safetensors", "safetensors", "data I/O"),
    ("einops", "einops", "model"),
    ("timm", "timm", "model"),
    ("omegaconf", "omegaconf", "config"),
    ("numpy", "numpy", "core"),
    ("PIL", "pillow", "preprocessing"),
    ("tqdm", "tqdm", "progress bars"),
    ("packaging", "packaging", "version checks"),
    ("triton", "triton", "flash attention backend"),
]

# ── Required models ───────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "RADIO v2.5-h (image encoder for REPA loss)",
        "path": "checkpoints/radio_v2.5-h.pth.tar",
        "url": "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-h.pth.tar",
        "required_for": "training",
    },
    {
        "name": "DC-AE VAE (image encoder for latent preprocessing)",
        "hf_model_id": "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        "required_for": "preprocessing",
    },
]


def check_packages():
    print("=" * 60)
    print("CHECKING PYTHON PACKAGES")
    print("=" * 60)

    missing = []
    for module_name, pip_name, required_for in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK]   {pip_name:<20s} {version:<15s} ({required_for})")
        except ImportError:
            print(f"  [MISS] {pip_name:<20s} {'—':<15s} ({required_for})")
            missing.append(pip_name)

    # Check flash_attn (optional — PyTorch SDPA fallback is available)
    try:
        from flash_attn import flash_attn_varlen_func
        print(f"  [OK]   {'flash-attn (optional)':<20s} {'available':<15s} (faster attention)")
    except ImportError:
        print(f"  [INFO] {'flash-attn (optional)':<20s} {'—':<15s} (using PyTorch SDPA fallback)")

    print()
    if missing:
        print(f"Missing {len(missing)} package(s). Install with:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("All packages installed.")
    print()
    return len(missing) == 0


def check_torch_details():
    print("=" * 60)
    print("TORCH ENVIRONMENT")
    print("=" * 60)
    try:
        import torch
        print(f"  PyTorch:        {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version:   {torch.version.cuda}")
            print(f"  GPU count:      {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}:          {name} ({mem:.1f} GB)")
        else:
            print("  WARNING: No CUDA GPU detected. Training requires GPU.")
    except ImportError:
        print("  PyTorch not installed.")
    print()


def check_models(download=True):
    print("=" * 60)
    print("CHECKING MODELS")
    print("=" * 60)

    all_ok = True

    for model in MODELS:
        if "path" in model:
            # File-based model (RADIO)
            exists = os.path.isfile(model["path"])

            if exists:
                print(f"  [OK]   {model['name']}")
                print(f"         Path: {model['path']}")
            else:
                print(f"  [MISS] {model['name']}")
                print(f"         Path: {model['path']}")
                if download:
                    print(f"         Downloading...")
                    os.makedirs(os.path.dirname(model["path"]), exist_ok=True)
                    result = subprocess.run(
                        ["wget", "-c", model["url"], "-O", model["path"]]
                    )
                    if result.returncode == 0 and os.path.isfile(model["path"]):
                        print(f"  [OK]   Downloaded successfully.")
                    else:
                        print(f"  [FAIL] Download FAILED.")
                        all_ok = False
                else:
                    all_ok = False

        elif "hf_model_id" in model:
            # HuggingFace model (DC-AE)
            cached = False
            hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_name = "models--" + model["hf_model_id"].replace("/", "--")
            cached = os.path.isdir(os.path.join(hf_cache, model_cache_name))

            if cached:
                print(f"  [OK]   {model['name']}")
                print(f"         HuggingFace ID: {model['hf_model_id']}")
            else:
                print(f"  [MISS] {model['name']}")
                print(f"         HuggingFace ID: {model['hf_model_id']}")
                if download:
                    print(f"         Downloading from HuggingFace...")
                    try:
                        from diffusers import AutoencoderDC
                        AutoencoderDC.from_pretrained(model["hf_model_id"])
                        print(f"  [OK]   Downloaded and cached successfully.")
                    except Exception as e:
                        print(f"  [FAIL] Download FAILED: {e}")
                        all_ok = False
                else:
                    all_ok = False

        print()

    if all_ok:
        print("All models available.")
    print()
    return all_ok


def check_nit_package():
    print("=" * 60)
    print("CHECKING NIT PACKAGE")
    print("=" * 60)
    try:
        import nit
        print("  [OK]   nit package is installed")
    except ImportError:
        print("  [MISS] nit package not found. Run: pip install -e .")
        print()
        return False

    # Check key submodules
    submodules = [
        ("nit.models.c2i.nit_model", "NiT model"),
        ("nit.schedulers.flow_matching.loss", "FlowMatchingLoss"),
        ("nit.data.packed_c2i_data", "data loader"),
        ("nit.data.pack", "packing algorithms"),
        ("nit.models.nvidia_radio.hubconf", "RADIO encoder"),
    ]
    all_ok = True
    for mod_name, desc in submodules:
        try:
            importlib.import_module(mod_name)
            print(f"  [OK]   {mod_name:<45s} ({desc})")
        except ImportError as e:
            print(f"  [FAIL] {mod_name:<45s} ({desc}): {e}")
            all_ok = False

    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Check environment and download models for NiT training."
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Only check, do not download missing models.",
    )
    args = parser.parse_args()

    download = not args.no_download

    print()
    pkg_ok = check_packages()
    check_torch_details()
    nit_ok = check_nit_package()
    models_ok = check_models(download=download)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = pkg_ok and nit_ok and models_ok
    if all_ok:
        print("  Everything is ready. You can start preprocessing and training.")
    else:
        if not pkg_ok:
            print("  - Some Python packages are missing.")
        if not nit_ok:
            print("  - NiT package or submodules not importable.")
        if not models_ok:
            print("  - Some models are missing.")
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
