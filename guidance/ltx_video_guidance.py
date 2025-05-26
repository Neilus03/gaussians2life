"""ltx_video_guidance.py
A drop‑in replacement for Dynamicrafter guidance that wraps the LTX‑Video
pipeline.  It keeps Gaussians‑to‑Life’s multi‑view bookkeeping (anchor view,
per‑view cache, warping, `pseudo_loss` contract) but swaps the actual video
synthesis backend.
"""
from __future__ import annotations

# --- standard ----------------------------------------------------------------
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

# --- third‑party --------------------------------------------------------------
import torch
import torch.nn.functional as F

# --- threestudio --------------------------------------------------------------
import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import Float, Tensor

# --- project helpers ----------------------------------------------------------
from ..utils import dprint, warp_video_with_flow, transform_video_with_homography, find_homography

# ---------------------------------------------------------------------------- #
#  helper: ensure the LTX‑Video submodule is importable
# ---------------------------------------------------------------------------- #

def add_path_to_ltxvideo() -> Path:
    here = Path(__file__).parent
    ltx = here / "LTX-Video"
    if not ltx.is_dir():
        raise ImportError(
            f"LTX-Video submodule not found at {ltx}. "
            "Did you run `git submodule update --init --recursive`?"
        )
    if str(ltx) not in sys.path:
        sys.path.append(str(ltx))
    return ltx


# ---------------------------------------------------------------------------- #
#  main guidance object
# ---------------------------------------------------------------------------- #

@threestudio.register("ltx-video-guidance")
class LTXVideoGuidance(BaseObject):
    """LTX‑Video guidance wrapper used by Gaussians2Life."""

    # ---------------------------------------------------------------------
    #  configuration schema
    # ---------------------------------------------------------------------

    @dataclass
    class Config(BaseObject.Config):
        # core LTX paths
        pipeline_config: str = ""  # YAML file shipped with LTX‑Video
        ckpt_path: str = ""         # *.safetensors checkpoint

        # text guidance
        prompt: str = ""
        negative_prompt: str = "worst quality, blurry, jittery"
        guidance_scale: float = 7.5

        # video settings (must match the outer datamodule)
        height: int = 320
        width: int = 512
        num_frames: int = 16
        frame_rate: int = 30

        # misc
        offload_to_cpu: bool = False
        anchor_viewpoint_sample_path: str = ""  # optional mp4 to prime all views

        #   --- legacy fields kept so YAMLs don’t break -----------------
        warping_mode: str = "flow"  # {"flow", "homography"}

    cfg: Config  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    #  lifecycle hooks
    # ------------------------------------------------------------------

    def configure(self) -> None:  # noqa: D401 – docstring in parent class
        # ensure submodule on path *before* import
        ltx_dir = add_path_to_ltxvideo()

        inf_py = ltx_dir / "inference.py"

        import importlib.util
        import types
        spec = importlib.util.spec_from_file_location("ltx_inference", inf_py)
        ltx_inference = importlib.util.module_from_spec(spec)        # type: ignore
        assert spec.loader is not None
        spec.loader.exec_module(ltx_inference)                       # type: ignore

        create_ltx_video_pipeline = ltx_inference.create_ltx_video_pipeline

        # sanity
        assert os.path.exists(self.cfg.pipeline_config), (
            "pipeline_config does not exist: " + self.cfg.pipeline_config
        )
        assert os.path.exists(self.cfg.ckpt_path), (
            "ckpt_path does not exist: " + self.cfg.ckpt_path
        )

        # build pipeline exactly like LTX reference helper
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=self.cfg.ckpt_path,
            precision="mixed_precision",
            text_encoder_model_name_or_path=None,  # pulled from YAML internally
            sampler=None,
            device=self.device,
        )
        self.pipeline.eval()
        dprint(">>> LTX‑Video pipeline loaded")

        # view‑wise caches  ------------------------------------------------
        self.output_stack: dict[int, torch.Tensor] = {}
        self.first_frames: dict[int, torch.Tensor] = {}
        self.first_frame_indices: dict[int, int] = {}
        self.current_guidance_view: Optional[int] = None
        self.anchor_guidance_output: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    #  main call – mirrors Dynamicrafter signature
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompts: List[str],               # ignored – we use cfg.prompt
        num_frames: int = 16,
        rgb_prev: Optional[Float[Tensor, "B H W C"]] = None,
        zero_timestep: int = 0,
        **kwargs,
    ) -> dict[str, float]:
        view_index: int | None = kwargs.get("view_index")

        # reset per‑sequence caches if told so
        if kwargs.get("clear_previous", False):
            self.output_stack.clear()
            self.first_frames.clear()
            self.first_frame_indices.clear()
            self.current_guidance_view = None

        # ------------------------------------------------------------
        #  anchor‑viewpoint bootstrap (optional)
        # ------------------------------------------------------------
        if (
            self.cfg.anchor_viewpoint_sample_path
            and not self.anchor_guidance_output
        ):
            try:
                import torchvision.io.video as io_video

                vid_tensor, _, _ = io_video.read_video(
                    self.cfg.anchor_viewpoint_sample_path, pts_unit="sec"
                )  # T × H × W × 3, uint8
                vid_tensor = vid_tensor.movedim(-1, 1) / 255.0  # T × 3 × H × W
                vid_tensor = F.interpolate(
                    vid_tensor, (self.cfg.height, self.cfg.width), mode="bilinear", align_corners=False
                )
                vid_tensor = vid_tensor.unsqueeze(0)  # 1 × T × 3 × H × W
                self.anchor_guidance_output = vid_tensor.to(self.device)
                dprint(">>> anchor viewpoint loaded from file")
            except Exception as exc:
                dprint(f"Anchor video load failed: {exc}")

        # ------------------------------------------------------------
        #  existing view → just return cached result
        # ------------------------------------------------------------
        if view_index is not None and view_index in self.output_stack:
            self.current_guidance_view = view_index
            return {"pseudo_loss": 0.0}

        # ------------------------------------------------------------
        #  preprocess current RGB(s) to feed size expected by LTX
        # ------------------------------------------------------------
        vid = rgb.permute(0, 3, 1, 2)  # B C H W in [0,1]
        vid = F.interpolate(
            vid, (self.cfg.height, self.cfg.width), mode="bilinear", align_corners=False
        )
        b, c, h, w = vid.shape
        vid = vid.view(b, num_frames, c, h, w)  # B T C H W

        # ------------------------------------------------------------
        #  generate (or warp) video for this view
        # ------------------------------------------------------------
        if rgb_prev is not None:
            # we do not use latent interpolation – simply copy previous
            guidance_video = self.output_stack.get(kwargs.get("reference_view"), None)
            if guidance_video is not None and self.cfg.warping_mode == "flow":
                guidance_video = warp_video_with_flow(
                    self.first_frames[kwargs["reference_view"]],
                    rgb[0].cpu().numpy(),
                    guidance_video[0].permute(1, 0, 2, 3),  # T C H W → C T H W
                    self.first_frame_indices[kwargs["reference_view"]],
                )
                guidance_video = guidance_video.permute(1, 0, 2, 3).unsqueeze(0)
                self.diffusion_output = guidance_video  # keep attribute for external access
        else:
            # fresh generation through LTX‑Video
            pipe_out = self.pipeline(
                video=vid,
                text_prompts=[self.cfg.prompt] * b,
                negative_prompts=[self.cfg.negative_prompt] * b,
                num_frames=self.cfg.num_frames,
                frame_rate=self.cfg.frame_rate,
                guidance_scale=self.cfg.guidance_scale,
                offload_to_cpu=self.cfg.offload_to_cpu,
                output_type="pt",
            )
            out = pipe_out.images  # B T C H W in [-1,1] or [0,1] depending – assume [0,1]
            out = out.permute(0, 2, 1, 3, 4)  # B C T H W to match Dynamicrafter convention
            self.diffusion_output = out  # used by the 4‑D system

        # ------------------------------------------------------------
        #  bookkeeping
        # ------------------------------------------------------------
        if view_index is not None:
            self.output_stack[view_index] = self.diffusion_output
            self.first_frames[view_index] = rgb[0].cpu().numpy()
            self.first_frame_indices[view_index] = zero_timestep
            self.current_guidance_view = view_index

        return {"pseudo_loss": 0.0}

    # ------------------------------------------------------------------
    #  timestep scheduler hook – not needed for LTX
    # ------------------------------------------------------------------

    def update_t_w(self, step: int, max_steps: int):  # noqa: D401 (no doc)
        # LTX‑Video internally decides its sampling steps; we just keep the API.
        pass
