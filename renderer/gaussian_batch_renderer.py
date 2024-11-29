import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

from ..geometry.gaussian_4d_motion import BasicPointCloud, Camera


class GaussianBatchRenderer:
    def batch_forward(self, batch):
        bs = batch["c2w"].shape[0]
        renders = []
        rendered_depths = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        masks = []

        _2d_means = []
        _2d_covs = []
        _2d_cov_invs = []
        _indices = []
        _weights = []
        _x_mus = []

        changes = []
        updated_vars = []

        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            if "fovx" not in batch:
                fovx = batch["fovy"][batch_idx]
            else:
                fovx = batch["fovx"][batch_idx]
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovx, fovy=fovy, znear=batch.get("znear", 0.1),
                zfar=batch.get("zfar", 100)
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovx,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam, self.background_tensor, **batch
                )
                renders.append(render_pkg["render"])
                rendered_depths.append(render_pkg["rendered_depth"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])

                _2d_means.append(render_pkg["2d_mean"])
                _2d_covs.append(render_pkg["2d_cov"])
                _2d_cov_invs.append(render_pkg["2d_cov_inv"])
                _indices.append(render_pkg["indices"])
                _weights.append(render_pkg["weights"])
                _x_mus.append(render_pkg["x_mu"])

                changes.append(render_pkg["changes"])
                updated_vars.append(render_pkg["updated_vars"])

                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if (
                    render_pkg.__contains__("pred_normal")
                    and render_pkg["pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["pred_normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "2d_mean": torch.stack(_2d_means, dim=0),
            "2d_cov": torch.stack(_2d_covs, dim=0),
            "2d_cov_inv": torch.stack(_2d_cov_invs, dim=0),
            "indices": torch.stack(_indices, dim=0),
            "weights": torch.stack(_weights, dim=0),
            "x_mu": torch.stack(_x_mus, dim=0),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
            "rendered_depth": torch.stack(rendered_depths, dim=0),
        }
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )

        if len(changes) == 1:
            changes_out = changes[0]
        else:
            changes_out = {}
            for key in changes[0].keys():
                changes_out[key] = torch.stack([change[key] for change in changes], dim=0)
        outputs.update({"changes": changes_out})

        if len(updated_vars) == 1:
            updated_vars_out = updated_vars[0]
        else:
            updated_vars_out = {}
            for key in updated_vars[0].keys():
                updated_vars_out[key] = torch.stack([updated_var[key] for updated_var in updated_vars], dim=0)
        outputs.update({"updated_vars": updated_vars_out})

        return outputs
