import logging
import os
import sys
from typing import Optional, Union

import torch
from diffusers import DDIMScheduler, LCMScheduler
from PIL import Image

# -----------------------------------------------------------------------------
# Make   /Marigold/src  import-able so that `models.qat_models` resolves.
# -----------------------------------------------------------------------------
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.append(_src_dir)

from models.qat_models import QAT_UNet, QAT_VAE, prepare_qat  # noqa: E402
from .marigold_depth_pipeline import (
    MarigoldDepthPipeline,  # re-use the heavy lifting from original pipeline
    MarigoldDepthOutput,
)

__all__ = ["QuantDepthPipeline", "create_default_quant_depth_pipeline"]


class QuantDepthPipeline(MarigoldDepthPipeline):
    """Marigold depth pipeline **without** CLIP / tokenizer and with QAT-ready
    lightweight UNet + VAE.  All logic (encode/decode, scheduler, ensembling)
    is inherited from `MarigoldDepthPipeline`.  We only override the parts that
    expected text-encoder inputs.
    """

    def __init__(
        self,
        unet: QAT_UNet,
        vae: QAT_VAE,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ) -> None:
        super().__init__(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        logging.info("QuantDepthPipeline initialised: CLIP/text-encoder removed, QAT-ready modules registered.")

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def encode_empty_text(self):
        """Instead of calling CLIP we just cache a zeros tensor with the correct
        shape `[1, 77, 1024]` and dtype/device matching the UNet.
        """
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        self.empty_text_embed = torch.zeros(1, 77, 1024, device=device, dtype=dtype)
        return self.empty_text_embed


# -----------------------------------------------------------------------------
# Utility constructor that returns a pipeline with default QAT-prepared modules
# -----------------------------------------------------------------------------

def create_default_quant_depth_pipeline(prepare_modules_for_qat: bool = True) -> QuantDepthPipeline:
    """Helper to build a ready-to-train pipeline in two lines.

    Example:
        pipe = create_default_quant_depth_pipeline()
        out  = pipe(Image.open("rgb.jpg"))
    """

    unet = QAT_UNet()
    vae = QAT_VAE()
    if prepare_modules_for_qat:
        unet = prepare_qat(unet)
        vae = prepare_qat(vae)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=True,
        set_alpha_to_one=False,
    )

    pipe = QuantDepthPipeline(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        default_denoising_steps=20,
        default_processing_resolution=512,
    )
    return pipe
