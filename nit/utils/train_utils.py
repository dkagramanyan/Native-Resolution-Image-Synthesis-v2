import os
import torch
import functools
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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def log_validation(
    model, accelerator, model_config, sample_dir, global_steps,
    num_samples_per_class=4, image_sizes=(256, 512, 1024),
    num_steps=50, cfg_scale=1.5,
):
    """Generate and save sample images at multiple resolutions during training.

    Args:
        model: the NiT model (accelerator-wrapped)
        accelerator: the Accelerator instance
        model_config: model config from OmegaConf
        sample_dir: directory to save generated images
        global_steps: current training step
        num_samples_per_class: number of images to generate per class per resolution
        image_sizes: tuple of output resolutions to generate
        num_steps: number of ODE sampling steps
        cfg_scale: classifier-free guidance scale
    """
    from nit.utils.model_utils import dc_ae_decode
    from diffusers import AutoencoderDC
    from PIL import Image

    if not accelerator.is_main_process:
        return

    # Unwrap model for inference
    unwrapped_model = accelerator.unwrap_model(model)
    was_training = unwrapped_model.training
    unwrapped_model.eval()

    device = accelerator.device
    dtype = torch.bfloat16
    num_classes = model_config.network.params.num_classes
    patch_size = model_config.network.params.patch_size
    spatial_downsample = 32  # DC-AE

    # Load VAE for decoding (only when needed, then discard to save memory)
    vae = AutoencoderDC.from_pretrained(model_config.vae_dir).to(device, dtype=torch.float32)
    vae.eval()

    step_dir = os.path.join(sample_dir, f"step_{global_steps:07d}")
    os.makedirs(step_dir, exist_ok=True)

    total_saved = 0

    for image_size in image_sizes:
        latent_h = image_size // spatial_downsample
        latent_w = image_size // spatial_downsample
        res_label = f"{image_size}x{image_size}"
        res_dir = os.path.join(step_dir, res_label)
        os.makedirs(res_dir, exist_ok=True)

        all_images = []

        for class_idx in range(num_classes):
            n = num_samples_per_class

            # Random latent noise
            z = torch.randn(
                n * latent_h * latent_w, unwrapped_model.in_channels, patch_size, patch_size,
                device=device, dtype=dtype
            )

            # Class labels
            y = torch.full((n,), class_idx, device=device, dtype=torch.long)

            # Spatial dimensions
            hw_list = torch.tensor(
                [[latent_h, latent_w]] * n, device=device, dtype=torch.int
            )

            # Null class for CFG
            y_null = torch.full((n,), num_classes, device=device, dtype=torch.long)
            samples = _euler_sampler_with_null(
                unwrapped_model, z, y, y_null, hw_list,
                num_steps=num_steps, cfg_scale=cfg_scale, dtype=dtype,
            )

            # Reshape from packed to spatial
            samples = rearrange(
                samples.to(torch.float32),
                '(b h w) c p1 p2 -> b c (h p1) (w p2)',
                b=n, h=latent_h, w=latent_w,
            )

            # Decode with VAE
            images = dc_ae_decode(vae, samples)
            images = ((images + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy()

            for i, img in enumerate(images):
                pil_img = Image.fromarray(img)
                pil_img.save(os.path.join(res_dir, f"class{class_idx:02d}_{i:02d}.png"))
                all_images.append(img)
                total_saved += 1

        # Save grid per resolution
        _save_grid(all_images, num_classes, num_samples_per_class,
                   os.path.join(step_dir, f"grid_{res_label}.png"))

    # Cleanup VAE to free GPU memory
    del vae
    torch.cuda.empty_cache()

    # Restore training mode
    if was_training:
        unwrapped_model.train()

    logger.info(f"Saved {total_saved} validation images to {step_dir}")


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
    import numpy as np

    if not images:
        return

    h, w = images[0].shape[:2]
    grid = np.zeros((num_classes * h, num_per_class * w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // num_per_class
        col = idx % num_per_class
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    PILImage.fromarray(grid).save(save_path)
