"""ltx_video_guidance.py
A drop‑in replacement for Dynamicrafter guidance that wraps the LTX‑Video
pipeline.  It keeps Gaussians‑to‑Life's multi‑view bookkeeping (anchor view,
per‑view cache, warping, `pseudo_loss` contract) but swaps the actual video
synthesis backend.
"""
from __future__ import annotations
import yaml

# --- standard ----------------------------------------------------------------
from dataclasses import dataclass, field
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union 
import inspect # Ensure this is imported

# --- third‑party --------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np 

# --- threestudio --------------------------------------------------------------
import threestudio # For threestudio.warn
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import Float, Tensor
from threestudio.utils.misc import get_device 

# --- project helpers ----------------------------------------------------------
# from ..utils import dprint 
def dprint(*args, **kwargs): # Placeholder
    print("[DEBUG]", *args, **kwargs)

import torchvision.transforms.functional as TF
from PIL import Image
import imageio

# ---------------------------------------------------------------------------- #
#  Helper: Ensure the LTX‑Video submodule is importable
# ---------------------------------------------------------------------------- #
LTX_VIDEO_REPO_PATH: Optional[Path] = None

def add_path_to_ltxvideo() -> Path:
    global LTX_VIDEO_REPO_PATH
    if LTX_VIDEO_REPO_PATH is not None:
        return LTX_VIDEO_REPO_PATH
    
    here = Path(__file__).parent.resolve() 
    ltx_repo_path = here / "LTX-Video"
    if not ltx_repo_path.is_dir():
        raise ImportError(
            f"LTX-Video submodule not found at {ltx_repo_path}. "
            "Ensure `git submodule update --init --recursive` was run "
            "and the LTX-Video folder exists relative to this guidance script."
        )
    if str(ltx_repo_path) not in sys.path:
        sys.path.insert(0, str(ltx_repo_path))
    LTX_VIDEO_REPO_PATH = ltx_repo_path
    dprint(f"Added LTX-Video repository to sys.path: {LTX_VIDEO_REPO_PATH}")
    return ltx_repo_path

add_path_to_ltxvideo() # Call at import time

# --- LTX-Video Imports (after sys.path is set) ---
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXVideoPipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

# ---------------------------------------------------------------------------- #
#  Main Guidance Object
# ---------------------------------------------------------------------------- #

@threestudio.register("ltx-video-guidance")
class LTXVideoGuidance(BaseObject):
    """LTX‑Video guidance wrapper used by Gaussians2Life."""

    @dataclass
    class Config(BaseObject.Config):
        pipeline_config: str 
        ckpt_path: str       
        
        prompt: str = ""     
        negative_prompt: str = "worst quality, low quality, blurry, inconsistent motion, jittery, distorted, text, watermark, signature"
        guidance_scale: float = 3.5 

        height: int = 320   
        width: int = 512    
        num_frames: int = 17 # TOTAL frames LTX generates (e.g., 1 cond + 16 new = 8*2+1 for LTX-2B)
        
        frame_rate: int = 24 
        seed: int = 42       

        image_cond_noise_scale: float = 0.025 
        skip_layer_strategy_str: str = "AttentionValues" 
        
        offload_to_cpu: bool = False 
        anchor_viewpoint_sample_path: str = "" 

        lambda_latent_interpolation: Any = 0.0 
        latent_interpolation_type: str = "previous"

        warping_mode: str = "flow" 

    cfg: Config
    pipeline: LTXVideoPipeline
    ltx_inference_module: Any 
    current_lambda_latent_interpolation: float = 0.0

    def configure(self) -> None:
        self.device = get_device()
        dprint(f"LTXVideoGuidance configured to use device: {self.device}")
        
        inf_py_path = LTX_VIDEO_REPO_PATH / "inference.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("ltx_inference_module", str(inf_py_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {inf_py_path}")
        self.ltx_inference_module = importlib.util.module_from_spec(spec)
        sys.modules["ltx_inference_module"] = self.ltx_inference_module 
        spec.loader.exec_module(self.ltx_inference_module)
        
        create_ltx_video_pipeline_func = self.ltx_inference_module.create_ltx_video_pipeline

        project_root = Path(os.getcwd()) 
        resolved_pipeline_config = project_root / self.cfg.pipeline_config
        resolved_ckpt_path = project_root / self.cfg.ckpt_path
        
        assert resolved_pipeline_config.is_file(), f"LTX pipeline_config not found: {resolved_pipeline_config}"
        assert resolved_ckpt_path.is_file(), f"LTX ckpt_path not found: {resolved_ckpt_path}"
        
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
        
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            dprint("LTX: VAE tiling enabled on pipeline object.")
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
            dprint("LTX: Progress bar disabled on pipeline object.")
        
        for component_name in ["transformer", "vae", "text_encoder"]:
            component = getattr(self.pipeline, component_name, None)
            if component is not None and hasattr(component, "eval"):
                component.eval()
                dprint(f"LTX: {component_name} set to eval mode.")

        dprint(f">>> LTX‑Video pipeline loaded and configured for device {self.device}")

        self.output_stack: dict[int, torch.Tensor] = {}
        self.first_frames: dict[int, np.ndarray] = {} 
        self.first_frame_indices: dict[int, int] = {}  
        self.current_guidance_view: Optional[int] = None
        self.anchor_guidance_output: Optional[torch.Tensor] = None 
        self.diffusion_output: Optional[torch.Tensor] = None       
        
        self.update_t_w(0, getattr(self.cfg, 'max_steps', 1))

    def _preprocess_pil_for_ltx_conditioning(self, pil_image: Image.Image) -> torch.Tensor:
        ltx_load_image_func = self.ltx_inference_module.load_image_to_tensor_with_resize_and_crop
        
        processed_tensor = ltx_load_image_func(
            pil_image,
            target_height=self.cfg.height, 
            target_width=self.cfg.width,   
            just_crop=True 
        )
        return processed_tensor.to(self.device)

    def _postprocess_ltx_output(self, generated_video_bcthw: torch.Tensor) -> torch.Tensor:
        final_h, final_w = self.cfg.height, self.cfg.width
        gen_h, gen_w = generated_video_bcthw.shape[-2:]

        if gen_h == final_h and gen_w == final_w:
            return generated_video_bcthw

        ltx_calculate_padding_func = self.ltx_inference_module.calculate_padding
        h_cfg_padded_target = ((self.cfg.height - 1) // 32 + 1) * 32
        w_cfg_padded_target = ((self.cfg.width - 1) // 32 + 1) * 32

        if gen_h == h_cfg_padded_target and gen_w == w_cfg_padded_target and \
           (self.cfg.height != h_cfg_padded_target or self.cfg.width != w_cfg_padded_target):
            padding_applied = ltx_calculate_padding_func(self.cfg.height, self.cfg.width, h_cfg_padded_target, w_cfg_padded_target)
            pad_left, pad_right, pad_top, pad_bottom = padding_applied
            
            crop_h_end = h_cfg_padded_target - pad_bottom
            crop_w_end = w_cfg_padded_target - pad_right
            if pad_bottom == 0: crop_h_end = h_cfg_padded_target 
            if pad_right == 0: crop_w_end = w_cfg_padded_target
            
            pad_top = max(0, pad_top); pad_left = max(0, pad_left)
            crop_h_end = min(h_cfg_padded_target, crop_h_end); crop_w_end = min(w_cfg_padded_target, crop_w_end)

            if pad_top >= crop_h_end or pad_left >= crop_w_end:
                dprint(f"LTX: Invalid crop dimensions. Resizing instead.")
                return F.interpolate(generated_video_bcthw, size=(final_h, final_w), mode='bilinear', align_corners=False)

            cropped_video = generated_video_bcthw[..., pad_top:crop_h_end, pad_left:crop_w_end]
            dprint(f"LTX: Cropped output from {gen_h}x{gen_w} to {cropped_video.shape[-2:]}")
            
            if cropped_video.shape[-2] != final_h or cropped_video.shape[-1] != final_w:
                dprint(f"LTX: Post-crop resize from {cropped_video.shape[-2:]} to {final_h}x{final_w}")
                cropped_video = F.interpolate(cropped_video, size=(final_h, final_w), mode='bilinear', align_corners=False)
            return cropped_video
        else:
            dprint(f"LTX: Output shape {gen_h}x{gen_w} unexpected. Resizing to {final_h}x{final_w}.")
            return F.interpolate(generated_video_bcthw, size=(final_h, final_w), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def __call__(
        self,
        rgb: Float[Tensor, "T_g2l H W C"], 
        prompts: List[str],
        num_frames: int, # G2L's system.geometry.num_frames
        rgb_prev: Optional[Float[Tensor, "B H W C"]] = None,
        zero_timestep: int = 0,
        **kwargs,
    ) -> dict[str, float]:
        view_index: int = kwargs.get("view_index", -1)

        if kwargs.get("clear_previous", False):
            self.output_stack.clear(); self.first_frames.clear(); self.first_frame_indices.clear()
            self.current_guidance_view = None; self.anchor_guidance_output = None; self.diffusion_output = None
            dprint("LTX: Cleared previous states on request.")

        if self.cfg.anchor_viewpoint_sample_path and self.anchor_guidance_output is None:
            try:
                pil_frames = []
                with imageio.get_reader(self.cfg.anchor_viewpoint_sample_path, 'ffmpeg') as reader:
                    # self.cfg.num_frames here refers to LTX's target output frames (e.g., 17)
                    for i, frame_np in enumerate(reader):
                        if i >= self.cfg.num_frames: break 
                        pil_img = Image.fromarray(frame_np).convert("RGB").resize((self.cfg.width, self.cfg.height))
                        pil_frames.append(TF.to_tensor(pil_img))
                
                if not pil_frames: raise ValueError("No frames from anchor video")
                
                T_anchor_loaded = len(pil_frames)
                T_ltx_target = self.cfg.num_frames # LTX target output length (e.g., 17)

                if T_anchor_loaded < T_ltx_target:
                    last_frame = pil_frames[-1]
                    pil_frames.extend([last_frame] * (T_ltx_target - T_anchor_loaded))
                elif T_anchor_loaded > T_ltx_target:
                    pil_frames = pil_frames[:T_ltx_target]

                stacked_frames = torch.stack(pil_frames, dim=0) 
                self.anchor_guidance_output = stacked_frames.permute(1,0,2,3).unsqueeze(0).to(self.device) 
                dprint(f">>> Anchor video loaded & processed to shape {self.anchor_guidance_output.shape} for LTX.")
            except Exception as exc:
                dprint(f"Anchor video load/process failed: {exc}. Anchor will not be used.")
                self.anchor_guidance_output = None
        
        use_anchor_for_view_0 = True 

        if view_index != -1: 
            if view_index in self.output_stack:
                self.current_guidance_view = view_index
                self.diffusion_output = self.output_stack[view_index]
                dprint(f"LTX: Reusing CACHED video for view_index {view_index}")
                return {"pseudo_loss": 0.0}
            elif use_anchor_for_view_0 and view_index == 0 and self.anchor_guidance_output is not None:
                self.diffusion_output = self.anchor_guidance_output
                self.output_stack[view_index] = self.diffusion_output
                g2l_static_frame_hwc_np = rgb[zero_timestep].detach().cpu().numpy()
                self.first_frames[view_index] = g2l_static_frame_hwc_np
                self.first_frame_indices[view_index] = 0 
                self.current_guidance_view = view_index
                dprint(f"LTX: Using loaded ANCHOR video for first view_index {view_index}.")
                return {"pseudo_loss": 0.0}

        dprint(f"LTX: Generating NEW video for view_index {view_index}")
        
        static_cond_frame_hwc_01 = rgb[zero_timestep].to(self.device) 
        cond_pil_image = TF.to_pil_image(static_cond_frame_hwc_01.permute(2,0,1).cpu()) 

        ltx_cond_input_tensor_m11 = self._preprocess_pil_for_ltx_conditioning(cond_pil_image)

        ltx_conditioning_item = ConditioningItem(
            media_item=ltx_cond_input_tensor_m11, 
            media_frame_number=0,                  
            conditioning_strength=1.0                    
        )

        text_prompt = self.cfg.prompt if self.cfg.prompt else prompts[0]
        current_negative_prompt = self.cfg.negative_prompt if self.cfg.negative_prompt else "" 
        
        effective_seed = self.cfg.seed + (view_index if view_index != -1 else os.getpid() % 10000)
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        
        stg_map = {"attentionvalues": SkipLayerStrategy.AttentionValues, "attentionskip": SkipLayerStrategy.AttentionSkip, "residual": SkipLayerStrategy.Residual, "transformerblock": SkipLayerStrategy.TransformerBlock}
        skip_strategy_enum = stg_map.get(self.cfg.skip_layer_strategy_str.lower().replace("_", ""), SkipLayerStrategy.AttentionValues)

        num_inference_steps_cfg = getattr(self.cfg, 'num_inference_steps', getattr(self.cfg, 'num_steps', 30))

        v2v_kwargs = {}
        source_video_for_v2v = None
        if self.current_lambda_latent_interpolation > 0.001: 
            if self.cfg.latent_interpolation_type == "previous" and \
               self.diffusion_output is not None and \
               (self.current_guidance_view is not None and self.current_guidance_view != view_index):
                source_video_for_v2v = self.diffusion_output
            elif self.cfg.latent_interpolation_type == "anchor" and self.anchor_guidance_output is not None:
                source_video_for_v2v = self.anchor_guidance_output
        
        if source_video_for_v2v is not None:
            if source_video_for_v2v.shape[2] == self.cfg.num_frames: 
                v2v_kwargs["image"] = source_video_for_v2v 
                v2v_kwargs["denoise_strength"] = 1.0 - self.current_lambda_latent_interpolation
                v2v_kwargs["denoise_strength"] = max(0.0, min(1.0, v2v_kwargs["denoise_strength"]))
                dprint(f"LTX: Using V2V-like with input shape {v2v_kwargs['image'].shape}, denoise_strength={v2v_kwargs['denoise_strength']:.3f}")
            else:
                dprint(f"LTX: V2V source video frame count mismatch. Source: {source_video_for_v2v.shape[2]}, Target LTX gen: {self.cfg.num_frames}. Skipping V2V.")

        pipe_kwargs = {
            "prompt": text_prompt,
            "negative_prompt": current_negative_prompt,
            "conditioning_items": [ltx_conditioning_item],
            "media_items": None, 
            "height": self.cfg.height,
            "width": self.cfg.width,
            "num_frames": self.cfg.num_frames, 
            "num_inference_steps": num_inference_steps_cfg,
            "guidance_scale": self.cfg.guidance_scale,
            "image_cond_noise_scale": self.cfg.image_cond_noise_scale,
            "frame_rate": self.cfg.frame_rate,
            "generator": generator,
            "output_type": "pt",
            "skip_layer_strategy": skip_strategy_enum,
            "vae_per_channel_normalize": True,
            **v2v_kwargs
        }
        
        pipeline_call_signature = inspect.signature(self.pipeline.__call__)
        if "offload_to_cpu" in pipeline_call_signature.parameters:
            pipe_kwargs["offload_to_cpu"] = self.cfg.offload_to_cpu
        if "mixed_precision" in pipeline_call_signature.parameters:
            if hasattr(self.pipeline, 'transformer') and (self.pipeline.transformer.dtype == torch.float16 or self.pipeline.transformer.dtype == torch.bfloat16):
                pipe_kwargs["mixed_precision"] = True
            else:
                pipe_kwargs["mixed_precision"] = False
        
        dprint(f"LTX CALL KWARGS for view {view_index}: {pipe_kwargs}") # Print all to 
    

        dprint(f"LTX CALL KWARGS for view {view_index}: H={pipe_kwargs['height']}, W={pipe_kwargs['width']}, NumFrames={pipe_kwargs['num_frames']}, Steps={pipe_kwargs['num_inference_steps']}")
        try:
            dprint(f"LTX: About to call pipeline. Explicit num_frames in pipe_kwargs: {pipe_kwargs.get('num_frames')}")
            dprint(f"LTX: self.cfg.num_frames (from YAML): {self.cfg.num_frames}")
            result = self.pipeline(**pipe_kwargs)
            dprint(f"LTX: Pipeline call returned result.images.shape: {result.images.shape}")
            generated_video_bcthw_01 = result.images.to(self.device, dtype=torch.float32)
        except Exception as e:
            # Using threestudio's warning logger for errors during pipeline call
            threestudio.warn(f"LTX Pipeline call failed for view {view_index}: {e}") # Corrected logging
            dprint(f"LTX Pipeline call FAILED for view {view_index}. Error: {type(e).__name__}: {e}")
            dummy_shape = (1, 3, self.cfg.num_frames, self.cfg.height, self.cfg.width)
            generated_video_bcthw_01 = torch.zeros(dummy_shape, device=self.device, dtype=torch.float32)
            self.diffusion_output = generated_video_bcthw_01.detach() # Ensure diffusion_output is set
            # return {"pseudo_loss": 0.0} # Allow G2L to continue with dummy output
            raise e # Re-raise to stop execution and see the full LTX traceback immediately

        processed_video_bcthw_01 = self._postprocess_ltx_output(generated_video_bcthw_01)
        self.diffusion_output = processed_video_bcthw_01.detach() 
        
        if view_index != -1:
            self.output_stack[view_index] = self.diffusion_output
            self.first_frames[view_index] = static_cond_frame_hwc_01.cpu().numpy() 
            self.first_frame_indices[view_index] = 0 
            self.current_guidance_view = view_index
        
        dprint(f"LTX: Generated video for view {view_index}, final shape: {self.diffusion_output.shape}")
        return {"pseudo_loss": 0.0}

    def update_t_w(self, step: int, max_steps: int):
        if hasattr(self.cfg, 'lambda_latent_interpolation'):
            if isinstance(self.cfg.lambda_latent_interpolation, (list, tuple)) and \
               len(self.cfg.lambda_latent_interpolation) == 2 and \
               max_steps > 0:
                start_val = float(self.cfg.lambda_latent_interpolation[0])
                end_val = float(self.cfg.lambda_latent_interpolation[1])
                progress = min(1.0, step / max_steps if max_steps > 0 else 0)
                self.current_lambda_latent_interpolation = start_val + (end_val - start_val) * progress
            elif isinstance(self.cfg.lambda_latent_interpolation, (float, int)):
                self.current_lambda_latent_interpolation = float(self.cfg.lambda_latent_interpolation)
            else: 
                self.current_lambda_latent_interpolation = 0.0
        else: 
            self.current_lambda_latent_interpolation = 0.0
        
        if step == 0 and (max_steps > 0 or max_steps == -1): # max_steps can be -1 for infinite
             dprint(f"LTX: Initialized/updated lambda_latent_interpolation to {self.current_lambda_latent_interpolation:.3f}")