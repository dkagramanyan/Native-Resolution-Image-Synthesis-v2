import os
import torch
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from diffusers.utils import logging
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def freeze_model(model, trainable_modules={}, verbose=False):
    logger.info("Start freeze")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if verbose:
            logger.info("freeze moduel: "+str(name))
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                if verbose:
                    logger.info("unfreeze moduel: "+str(name))
                break
    logger.info("End freeze")
    params_unfreeze = [p.numel() if p.requires_grad == True else 0 for n, p in model.named_parameters()]
    params_freeze = [p.numel() if p.requires_grad == False else 0 for n, p in model.named_parameters()]
    logger.info(f"Unfreeze Module Parameters: {sum(params_unfreeze) / 1e6} M")
    logger.info(f"Freeze Module Parameters: {sum(params_freeze) / 1e6} M")
    return


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(ema_model, 'module'):
        ema_model = ema_model.module
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def log_validation(
    model, accelerator, model_config, sample_dir, global_steps,
    num_samples_per_class=4, image_sizes=(256, 512, 1024),
    num_steps=50, cfg_scale=1.5,
    fid_num_samples=100, fid_real_image_dir=None,
):
    """Generate sample images at multiple resolutions and compute FID per resolution.

    Uses 3-phase approach to avoid OOM:
      Phase 1: Generate latents with NiT on GPU, store on CPU
      Phase 2: Move NiT to CPU, load VAE on GPU, decode and save images
      Phase 3: Compute FID per resolution, cleanup, move NiT back to GPU
    """
    from nit.utils.model_utils import dc_ae_decode
    from diffusers import AutoencoderDC
    from PIL import Image

    if not accelerator.is_main_process:
        return

    unwrapped_model = accelerator.unwrap_model(model)
    was_training = unwrapped_model.training
    unwrapped_model.eval()

    device = accelerator.device
    dtype = torch.bfloat16
    num_classes = model_config.network.params.num_classes
    patch_size = model_config.network.params.patch_size
    spatial_downsample = 32  # DC-AE

    step_dir = os.path.join(sample_dir, f"step_{global_steps:07d}")
    os.makedirs(step_dir, exist_ok=True)

    # ── Phase 1: Generate latents on GPU, store on CPU ──────────────────────
    # Per resolution: grid samples + FID samples
    all_latents = {}  # {image_size: [latent_cpu, ...]}

    for image_size in image_sizes:
        latent_h = image_size // spatial_downsample
        latent_w = image_size // spatial_downsample
        total_for_res = max(fid_num_samples, num_samples_per_class * num_classes)
        latents_list = []

        logger.info(f"Validation: generating {total_for_res} latents at {image_size}x{image_size}...")
        for i in range(total_for_res):
            class_idx = i % num_classes
            latent = _generate_one_sample(
                unwrapped_model, class_idx, num_classes,
                latent_h, latent_w, patch_size,
                num_steps, cfg_scale, dtype, device,
            )
            latents_list.append(latent.cpu())
        all_latents[image_size] = latents_list

    # ── Phase 2: Move NiT to CPU, load VAE, decode and save ────────────────
    unwrapped_model.cpu()
    torch.cuda.empty_cache()

    vae = AutoencoderDC.from_pretrained(model_config.vae_dir).to(device, dtype=torch.float32)
    vae.eval()

    total_saved = 0
    fid_images_per_res = {}  # {image_size: [numpy_img, ...]}

    for image_size in image_sizes:
        res_label = f"{image_size}x{image_size}"
        res_dir = os.path.join(step_dir, res_label)
        os.makedirs(res_dir, exist_ok=True)

        decoded_images = []
        logger.info(f"Validation: decoding {len(all_latents[image_size])} samples at {res_label}...")
        for i, latent in enumerate(all_latents[image_size]):
            img = _decode_latent_to_numpy(vae, latent, device)
            Image.fromarray(img).save(os.path.join(res_dir, f"{i:05d}.png"))
            decoded_images.append(img)
            total_saved += 1

        # Save grid from first num_samples_per_class * num_classes images
        grid_count = num_samples_per_class * num_classes
        grid_images = decoded_images[:grid_count]
        _save_grid(grid_images, num_classes, num_samples_per_class,
                   os.path.join(step_dir, f"grid_{res_label}.png"))

        fid_images_per_res[image_size] = decoded_images

    del vae, all_latents
    torch.cuda.empty_cache()

    # ── Phase 3: Compute FID per resolution, move NiT back ──────────────────
    for image_size in image_sizes:
        res_label = f"{image_size}x{image_size}"
        fid_value = _compute_fid(
            fid_images_per_res[image_size], fid_real_image_dir, device,
            resize_to=299,
        )
        if fid_value is not None:
            logger.info(f"Validation step {global_steps}: FID-{res_label} = {fid_value:.4f}")
            accelerator.log({f"fid_{res_label}": fid_value}, step=global_steps)

    del fid_images_per_res
    torch.cuda.empty_cache()

    # Move NiT back to GPU
    unwrapped_model.to(device)

    if was_training:
        unwrapped_model.train()

    logger.info(f"Saved {total_saved} validation images to {step_dir}")


def _generate_one_sample(model, class_idx, num_classes, latent_h, latent_w,
                          patch_size, num_steps, cfg_scale, dtype, device):
    """Generate one latent sample (before VAE decoding)."""
    z = torch.randn(
        latent_h * latent_w, model.in_channels, patch_size, patch_size,
        device=device, dtype=dtype,
    )
    y = torch.tensor([class_idx], device=device, dtype=torch.long)
    hw_list = torch.tensor([[latent_h, latent_w]], device=device, dtype=torch.int)
    y_null = torch.tensor([num_classes], device=device, dtype=torch.long)

    sample = _euler_sampler_with_null(
        model, z, y, y_null, hw_list,
        num_steps=num_steps, cfg_scale=cfg_scale, dtype=dtype,
    )
    return rearrange(
        sample.to(torch.float32),
        '(h w) c p1 p2 -> 1 c (h p1) (w p2)',
        h=latent_h, w=latent_w,
    )


def _decode_latent_to_numpy(vae, latent, device):
    """Decode a single latent tensor to a uint8 numpy HWC image."""
    from nit.utils.model_utils import dc_ae_decode
    img = dc_ae_decode(vae, latent.to(device))
    img = ((img + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
    return img[0].permute(1, 2, 0).cpu().numpy()


def _compute_fid(generated_images, real_image_dir, device, resize_to=299):
    """Compute FID between generated images and real images using torchmetrics."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.warning("torchmetrics not installed, skipping FID. Install with: pip install torchmetrics[image]")
        return None

    if real_image_dir is None or not os.path.isdir(real_image_dir):
        logger.warning(f"Real image dir not found: {real_image_dir}, skipping FID.")
        return None

    from PIL import Image
    import torchvision.transforms.functional as TF

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Collect real image paths
    real_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(real_image_dir)
        for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if len(real_files) == 0:
        logger.warning(f"No images found in {real_image_dir}, skipping FID.")
        return None

    num_real = min(len(real_files), len(generated_images))
    batch_size = 32

    # Add real images
    for i in range(0, num_real, batch_size):
        batch_files = real_files[i:i + batch_size]
        batch = []
        for f in batch_files:
            img = Image.open(f).convert("RGB").resize((resize_to, resize_to))
            batch.append(TF.to_tensor(img))
        batch = torch.stack(batch).to(device)
        fid.update(batch, real=True)

    # Add generated images
    for i in range(0, len(generated_images), batch_size):
        batch_imgs = generated_images[i:i + batch_size]
        batch = []
        for img in batch_imgs:
            pil = Image.fromarray(img).resize((resize_to, resize_to))
            batch.append(TF.to_tensor(pil))
        batch = torch.stack(batch).to(device)
        fid.update(batch, real=False)

    fid_value = fid.compute().item()
    del fid
    return fid_value


def _euler_sampler_with_null(model, latents, y, y_null, hw_list,
                              num_steps=50, cfg_scale=1.5, dtype=torch.bfloat16):
    """ODE Euler sampler with proper null class for CFG."""
    _dtype = dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_cur = x_next
        if cfg_scale > 1.0:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
            hw_list_cur = torch.cat([hw_list, hw_list], dim=0)
        else:
            model_input = x_cur
            y_cur = y
            hw_list_cur = hw_list

        kwargs = dict(y=y_cur, hw_list=hw_list_cur)
        time_input = torch.ones(y_cur.size(0), device=device, dtype=torch.float64) * t_cur
        d_cur = model(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        ).to(torch.float64)

        if cfg_scale > 1.0:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next = x_cur + (t_next - t_cur) * d_cur

    return x_next


def _save_grid(images, num_classes, num_per_class, save_path):
    """Save a grid of images: rows=classes, cols=samples."""
    from PIL import Image as PILImage

    if not images:
        return

    h, w = images[0].shape[:2]
    grid = np.zeros((num_classes * h, num_per_class * w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // num_per_class
        col = idx % num_per_class
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    PILImage.fromarray(grid).save(save_path)
