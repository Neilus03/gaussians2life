# FILE: /home/stud/deln/storage/user/projects/threestudio-1/custom/gaussians2life/guidance/ltx_video_guidance.py

"""
ltx_video_guidance.py
A drop-in replacement for Dynamicrafter guidance that wraps the LTX-Video
pipeline. It preserves Gaussians-to-Life's multi-view workflow, including
the optional use of a pre-generated anchor video for bootstrapping, and uses a robust
file-based conditioning method to ensure data integrity.
"""
from __future__ import annotations
import yaml

# --- standard ----------------------------------------------------------------
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union 
import inspect
import tempfile

# --- third-party --------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np 

# --- threestudio --------------------------------------------------------------
import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import Float, Tensor
from threestudio.utils.misc import get_device 

# --- project helpers ----------------------------------------------------------
def dprint(*args, **kwargs):
    print("[DEBUG]", *args, **kwargs)

import torchvision.transforms.functional as TF
from PIL import Image
import imageio

# --- LTX-Video Imports ---
LTX_VIDEO_REPO_PATH: Optional[Path] = None

def add_path_to_ltxvideo() -> Path:
    global LTX_VIDEO_REPO_PATH
    if LTX_VIDEO_REPO_PATH is not None:
        return LTX_VIDEO_REPO_PATH
    
    here = Path(__file__).parent.resolve() 
    ltx_repo_path = here / "LTX-Video"
    if not ltx_repo_path.is_dir():
        raise ImportError(f"LTX-Video submodule not found at {ltx_repo_path}.")
    if str(ltx_repo_path) not in sys.path:
        sys.path.insert(0, str(ltx_repo_path))
    LTX_VIDEO_REPO_PATH = ltx_repo_path
    dprint(f"Added LTX-Video repository to sys.path: {LTX_VIDEO_REPO_PATH}")
    return ltx_repo_path

add_path_to_ltxvideo()

from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


@threestudio.register("ltx-video-guidance")
class LTXVideoGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # All parameters from the YAML are now correctly defined here
        pipeline_config: str 
        ckpt_path: str       
        prompt: str = ""     
        negative_prompt: str = "worst quality, low quality, blurry, inconsistent motion, jittery, distorted, text, watermark, signature"
        guidance_scale: float = 7.5
        height: int = 320   
        width: int = 512    
        num_frames: int = 32
        frame_rate: int = 16 
        seed: int = 42       
        image_cond_noise_scale: float = 0.01 
        skip_layer_strategy_str: str = "AttentionValues" 
        offload_to_cpu: bool = False
        anchor_viewpoint_sample_path: str = "" # FIX: This parameter is now correctly included
        lambda_latent_interpolation: Any = 0.0 
        latent_interpolation_type: str = "previous"
        warping_mode: str = "flow" 

    cfg: Config
    pipeline: LTXVideoPipeline
    ltx_inference_module: Any

    def configure(self) -> None:
        self.device = get_device()
        dprint(f"LTXVideoGuidance configured to use device: {self.device}")
        
        inf_py_path = LTX_VIDEO_REPO_PATH / "inference.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("ltx_inference_module", str(inf_py_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {inf_py_path}")
        self.ltx_inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ltx_inference_module)
        
        create_ltx_video_pipeline_func = self.ltx_inference_module.create_ltx_video_pipeline

        project_root = Path(os.getcwd()) 
        resolved_pipeline_config = project_root / self.cfg.pipeline_config
        resolved_ckpt_path = project_root / self.cfg.ckpt_path
        
        with open(resolved_pipeline_config, "r") as f:
            ltx_yaml_cfg = yaml.safe_load(f)

        self.pipeline = create_ltx_video_pipeline_func(
            ckpt_path=str(resolved_ckpt_path), 
            precision=ltx_yaml_cfg.get("precision", "fp16"),
            text_encoder_model_name_or_path=ltx_yaml_cfg.get("text_encoder_model_name_or_path"),
            sampler=ltx_yaml_cfg.get("sampler", "from_checkpoint"),
            device=self.device, 
            enhance_prompt=False, 
        )
        
        self.pipeline.set_progress_bar_config(disable=True)
        for component in ["transformer", "vae", "text_encoder"]:
            if hasattr(self.pipeline, component):
                getattr(self.pipeline, component).eval()
        dprint(f">>> LTX‑Video pipeline loaded and configured for device {self.device}")
        
        self.output_stack: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(
        self,
        rgb: Float[Tensor, "T_g2l H W C"], 
        prompts: List[str],
        num_frames: int,
        step: int,
        max_steps: int,
        zero_timestep: int = 0,
        **kwargs,
    ) -> dict[str, float]:
        
        view_index: int = kwargs.get("view_index", -1)

        if kwargs.get("clear_previous", False):
            self.output_stack.clear()

        if view_index in self.output_stack:
            self.diffusion_output = self.output_stack[view_index]
            dprint(f"LTX: Reusing CACHED video for view_index {view_index}")
            return {"pseudo_loss": 0.0}

        # FIX: Implement anchor video loading logic
        if view_index == 0 and self.cfg.anchor_viewpoint_sample_path:
            dprint(f"LTX: Loading anchor video from: {self.cfg.anchor_viewpoint_sample_path}")
            try:
                pil_frames = [
                    Image.fromarray(frame).convert("RGB").resize((self.cfg.width, self.cfg.height))
                    for frame in imageio.get_reader(self.cfg.anchor_viewpoint_sample_path, 'ffmpeg')
                ]
                if len(pil_frames) > num_frames:
                    pil_frames = pil_frames[:num_frames]
                
                video_tensor = torch.stack([TF.to_tensor(img) for img in pil_frames]) # T, C, H, W
                self.diffusion_output = video_tensor.unsqueeze(0).to(self.device) # B, T, C, H, W -> B, C, T, H, W for G2L
                self.output_stack[view_index] = self.diffusion_output
                dprint(f"Loaded and cached anchor video of shape {self.diffusion_output.shape}")
                return {"pseudo_loss": 0.0}
            except Exception as e:
                dprint(f"WARNING: Could not load anchor video, proceeding with generation. Error: {e}")

        dprint(f"LTX: guidance step {step}/{max_steps-1} for view_index {view_index}")
        
        # 1. Get the rendered frame from G2L and convert to a PIL Image.
        static_cond_frame = rgb[zero_timestep]
        conditioning_pil_image = TF.to_pil_image(static_cond_frame.permute(2, 0, 1).cpu())

        # 2. Create a temporary file to save the image.
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_image_path = temp_file.name
        
        try:
            conditioning_pil_image.save(temp_image_path)
            
            # 3. Dynamically anneal the noise scale.
            if max_steps > 1:
                progress = min(step / (max_steps - 1), 1.0)
                start_noise = 0.001
                current_noise_scale = start_noise + (self.cfg.image_cond_noise_scale - start_noise) * progress
            else:
                current_noise_scale = 0.001
            dprint(f"LTX: Annealed image_cond_noise_scale = {current_noise_scale:.4f}")

            # 4. Prepare pipeline arguments using the file path via the official helper.
            text_prompt = self.cfg.prompt if self.cfg.prompt else prompts[0]
            effective_seed = self.cfg.seed + view_index
            generator = torch.Generator(device=self.device).manual_seed(effective_seed)
            
            conditioning_items = self.ltx_inference_module.prepare_conditioning(
                conditioning_media_paths=[temp_image_path],
                conditioning_strengths=[1.0],
                conditioning_start_frames=[0],
                height=self.cfg.height,
                width=self.cfg.width,
                num_frames=num_frames,
                padding=(0,0,0,0),
                pipeline=self.pipeline
            )

            stg_map = {"attentionvalues": SkipLayerStrategy.AttentionValues, "attentionskip": SkipLayerStrategy.AttentionSkip}
            skip_strategy_enum = stg_map.get(self.cfg.skip_layer_strategy_str.lower().replace("_", ""), SkipLayerStrategy.AttentionValues)

            pipe_kwargs = {
                "prompt": text_prompt,
                "negative_prompt": self.cfg.negative_prompt,
                "conditioning_items": conditioning_items,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "num_frames": num_frames, 
                "num_inference_steps": 40,
                "guidance_scale": self.cfg.guidance_scale,
                "image_cond_noise_scale": current_noise_scale,
                "frame_rate": self.cfg.frame_rate,
                "generator": generator,
                "output_type": "pt",
                "skip_layer_strategy": skip_strategy_enum,
                "vae_per_channel_normalize": True, # Fix for the KeyError
            }
            
            # 5. Call the pipeline.
            result = self.pipeline(**pipe_kwargs)
            generated_video = result.images.to(self.device, dtype=torch.float32)

        finally:
            # 6. ALWAYS ensure the temporary file is deleted.
            temp_file.close()
            os.unlink(temp_image_path)
        
        # 7. Post-process and cache the output.
        if generated_video.shape[2] > num_frames:
            dprint(f"LTX generated {generated_video.shape[2]} frames, but G2L expects {num_frames}. Truncating.")
            generated_video = generated_video[:, :, :num_frames, :, :]

        self.diffusion_output = generated_video.detach() 
        
        if view_index != -1:
            self.output_stack[view_index] = self.diffusion_output
        
        dprint(f"LTX: Generated video for view {view_index}, final shape: {self.diffusion_output.shape}")
        return {"pseudo_loss": 0.0}