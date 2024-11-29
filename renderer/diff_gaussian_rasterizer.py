import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer
from ..geometry.utils import quat_mult


@threestudio.register("diff-gaussian-rasterizer-dynamic")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 0.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        time=None,
        time_step=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        assert time is not None, "time must be provided"

        invert_bg_color = np.random.rand() < self.cfg.invert_bg_prob

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc = self.geometry

        changes = pc.update(time, time_step=time_step)

        _xyz = pc.get_xyz.clone()
        if "add_rot" in changes:  # Apply rotation pre-activation
            _rotation = pc._rotation
        else:
            _rotation = pc.get_rotation
        _rotation = _rotation.clone()

        if "add_scale" in changes:  # Apply scale pre-activation
            _scaling = pc._scaling
        else:
            _scaling = pc.get_scaling
        _scaling = _scaling.clone()

        pos_change = changes["displacement"]
        _xyz += pos_change

        if "rotation" in changes:
            rot_change = changes["rotation"]
            if "add_rot" in changes:
                _rotation = _rotation + rot_change
            else:
                _rotation = quat_mult(_rotation, rot_change)
            _rotation = torch.nn.functional.normalize(_rotation, dim=-1)

        if "scale" in changes:
            scale_change = changes["scale"]
            if "add_scale" in changes:
                _scaling += scale_change
                _scaling = pc.apply_scaling_activation(_scaling)
            else:
                _scaling *= scale_change

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                _xyz, dtype=_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = _xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = _scaling
        rotations = _rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha, proj_means_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(0) + 1e-6)

        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image.clamp(0, 1),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "changes": changes,
            "updated_vars": {
                "xyz": _xyz,
                "rotation": _rotation,
                "scaling": _scaling,
            },
            "2d_mean": proj_means_2D,
            "2d_cov": conic_2D_inv,
            "2d_cov_inv": conic_2D,
            "indices": gs_per_pixel.long(),
            "weights": weight_per_gs_pixel,
            "x_mu": x_mu,
            "rendered_depth": rendered_depth,
        }
