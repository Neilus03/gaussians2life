from dataclasses import dataclass

import numpy as np
import threestudio
import torch
from plyfile import PlyData
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.typing import *

from .utils import o3d_knn, build_scaling_rotation, strip_symmetric, inverse_sigmoid
import math
from ..utils import dprint


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor


@threestudio.register("gaussian-splatting-trajectories")
class GaussianModel(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        sh_degree: int = 0

        geometry_convert_from: str = ""
        load_ply_only_vertex: bool = False
        mask_path: str = ""
        mask_box: Optional[List[List[float]]] = None
        box_rot_x: int = 0
        box_rot_y: int = 0
        box_rot_z: int = 0

        num_frames: int = 16
        update_scale: bool = False
        update_rotation: bool = True
        num_nearest_neighbors: int = 10
        num_nearest_neighbors_inference: int = 50
        knn_weighting: float = 4.0
        inference_mode: str = "displacement+rigid"
        anchor_view_multiplier: int = 5

        save_path_deformations: str = ""
        discard_mask: bool = False

    cfg: Config

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def configure(self) -> None:
        super().configure()
        self.active_sh_degree = 0
        self.max_sh_degree = self.cfg.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.movement_mask = None
        self.setup_functions()

        self.load_ply(self.cfg.geometry_convert_from)
        assert sum((self.cfg.mask_path != "",
                    self.cfg.mask_box is not None)) <= 1, "Cannot use more than one for 'mask_path', 'learnable_mask' and 'mask_box'."
        if self.cfg.mask_path != "":
            from .utils import transform_scene_from_bbox
            self.movement_mask = torch.load(self.cfg.mask_path)
            self._xyz, self._rotation, self._scaling = transform_scene_from_bbox(self._xyz,
                                                                                 self._rotation,
                                                                                 self._scaling,
                                                                                 self._xyz[self.movement_mask],
                                                                                 rot_x=self.cfg.box_rot_x,
                                                                                 rot_y=self.cfg.box_rot_y,
                                                                                 rot_z=self.cfg.box_rot_z)
            # self.mask_box = self._xyz[self.movement_mask].clone()
            import open3d as o3d

            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(self._xyz.cpu().numpy())).get_minimal_oriented_bounding_box()
            self.mask_box = torch.tensor(np.asarray(bbox.get_box_points()), dtype=torch.float32).to(self._xyz.device)


        elif self.cfg.mask_box is not None:
            from .utils import get_points_inside_bbox, transform_scene_from_bbox
            mask_box = torch.tensor(self.cfg.mask_box)
            # Compute mask from bounding box
            self.movement_mask = torch.tensor(get_points_inside_bbox(self._xyz, mask_box),
                                              dtype=torch.long)
            # Transform scene to align with bounding box
            self._xyz, self._rotation, self._scaling = transform_scene_from_bbox(self._xyz,
                                                                                 self._rotation,
                                                                                 self._scaling,
                                                                                 mask_box,
                                                                                 rot_x=self.cfg.box_rot_x,
                                                                                 rot_y=self.cfg.box_rot_y,
                                                                                 rot_z=self.cfg.box_rot_z)
            self.mask_box, _, _ = transform_scene_from_bbox(mask_box,
                                                            torch.zeros(1, 4),
                                                            torch.ones(1, 3),
                                                            mask_box,
                                                            rot_x=self.cfg.box_rot_x,
                                                            rot_y=self.cfg.box_rot_y,
                                                            rot_z=self.cfg.box_rot_z)

        if self.cfg.discard_mask:
            self.movement_mask = None
            self.mask_box = None

        self.trajectory_length = None
        self.timesteps = []
        self.extrinsics = []
        self.num_trajectories = []
        self.anchor_trajectories = None
        self.anchor_confidence = None
        self.nearest_anchor_index = None
        self.nearest_anchor_offset = None
        self.nearest_anchor_weight = None
        self.zero_timestep = 0

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def apply_scaling_activation(self, scaling):
        return self.scaling_activation(scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def apply_rotation_activation(self, rotation):
        return self.rotation_activation(rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def register_trajectories(self, trajectories, extrinsics, confidence, timestep=0):
        """
        Register trajectories of tracked points.

        :param trajectories: trajectories in camera space (T, N, 3)
        :param extrinsics: camera extrinsics (4, 4)
        :param confidence: uncertainties for depth estimation (T, N)
        """
        curr_index = self.anchor_trajectories.shape[1] if self.anchor_trajectories is not None else 0

        if self.trajectory_length is None:
            self.trajectory_length = trajectories.shape[0]
        else:
            assert self.trajectory_length == trajectories.shape[0], "Trajectory length mismatch"

        # filter trajectories with large std
        std_threshold_along_axis = 0.2
        trajectories = trajectories[:, trajectories.std(dim=0).max(dim=1).values < std_threshold_along_axis]

        with torch.no_grad():
            # change y and z axis
            rot_trajectories = trajectories.clone()
            rot_trajectories[..., 1] *= -1
            rot_trajectories[..., 2] *= -1
            rot_trajectories = rot_trajectories[timestep][None]
            rot_trajectories = torch.cat((rot_trajectories, torch.ones_like(rot_trajectories[..., :1])), dim=-1)
            rot_trajectories = rot_trajectories.permute(1, 0, 2)
            rot_trajectories = torch.bmm(rot_trajectories,
                                         extrinsics[None].repeat_interleave(len(rot_trajectories), dim=0).transpose(1,
                                                                                                                    2))
            rot_trajectories = rot_trajectories.permute(1, 0, 2)
            rot_trajectories = rot_trajectories[0, :, :3]

        # filter trajectories to only contain points in bounding box
        if hasattr(self, "mask_box") and self.mask_box is not None:
            from .utils import get_points_inside_bbox
            mask = get_points_inside_bbox(rot_trajectories, self.mask_box)
            if len(mask) == 0:
                print(timestep)
                torch.save(self.mask_box.cpu(), "mask_box.pt")
                torch.save(trajectories.cpu(), "trajectories.pt")
                print("No points of trajectories in bounding box.")
                return
            trajectories = trajectories[:, mask]
            rot_trajectories = rot_trajectories[mask]

        self.timesteps.append(timestep)
        self.extrinsics.append(extrinsics)
        self.num_trajectories.append(trajectories.shape[1])

        if self.anchor_trajectories is None:
            self.anchor_trajectories = trajectories
            self.anchor_confidence = confidence
        else:
            self.anchor_trajectories = torch.cat((self.anchor_trajectories, trajectories), dim=1)
            self.anchor_confidence = torch.cat((self.anchor_confidence, confidence), dim=1)

        _xyz = self._xyz
        if self.movement_mask is not None:
            _xyz = _xyz[self.movement_mask]
        knn_indices, knn_relative_positions, knn_squared_dists = o3d_knn(_xyz,
                                                                         self.cfg.num_nearest_neighbors if curr_index != 0 else self.cfg.num_nearest_neighbors * self.cfg.anchor_view_multiplier,
                                                                         rot_trajectories)

        if self.cfg.knn_weighting > 0:
            nearest_anchor_weight = torch.exp(-knn_squared_dists / self.cfg.knn_weighting)
        else:
            nearest_anchor_weight = torch.ones_like(knn_squared_dists)

        dprint(f"KNN mean squared dist: {knn_squared_dists.mean().item()}")
        knn_indices = knn_indices + curr_index
        if self.nearest_anchor_index is None:
            self.nearest_anchor_index = knn_indices
            self.nearest_anchor_offset = knn_relative_positions
            self.nearest_anchor_weight = nearest_anchor_weight
        else:
            self.nearest_anchor_index = torch.cat((self.nearest_anchor_index, knn_indices), dim=1)
            self.nearest_anchor_offset = torch.cat((self.nearest_anchor_offset, knn_relative_positions), dim=1)
            self.nearest_anchor_weight = torch.cat((self.nearest_anchor_weight, nearest_anchor_weight), dim=1)
        with torch.no_grad():
            self.infer_movement_from_trajectories()

    def infer_movement_from_trajectories(self):
        """
        Infer movement from registered trajectories.

        :param init_timestep: initial timestep for inference
        """
        _xyz = self._xyz
        if self.movement_mask is not None:
            _xyz = _xyz[self.movement_mask]

        animation_duration = max(self.timesteps) + self.trajectory_length - min(self.timesteps)

        displacement = torch.zeros(self._xyz.shape[0], animation_duration, 3, device=_xyz.device)
        rotation = torch.zeros(self._xyz.shape[0], animation_duration, 4, device=_xyz.device)
        rotation[..., 0] = 1
        scaling = torch.ones(self._xyz.shape[0], animation_duration, 3, device=_xyz.device)

        # transform anchor trajectories to world space (for now, naive)
        extrinsics = torch.cat([self.extrinsics[i][None].repeat_interleave(self.num_trajectories[i], 0)
                                for i in range(len(self.extrinsics))], dim=0)
        # change y and z axis
        points_3d_world = self.anchor_trajectories.clone()
        points_3d_world[..., 1] *= -1
        points_3d_world[..., 2] *= -1
        points_3d_world = torch.cat((points_3d_world, torch.ones_like(self.anchor_trajectories[..., :1])),
                                    dim=-1)
        points_3d_world = points_3d_world.permute(1, 0, 2)
        points_3d_world = torch.bmm(points_3d_world, extrinsics.transpose(1, 2))
        points_3d_world = points_3d_world.permute(1, 0, 2)
        points_3d_world = points_3d_world[..., :3]

        i = 0
        for t in range(- max(self.timesteps), self.trajectory_length - min(self.timesteps)):
            if t == 0:
                i += 1
                continue
            applicable_sequence_indices = [1 if 0 <= timestep + t < self.trajectory_length else 0
                                           for timestep in self.timesteps]
            applicable_trajectory_indices = torch.cat([
                torch.from_numpy(
                    np.repeat(applicable_sequence_indices[0],
                              self.cfg.num_nearest_neighbors * (self.cfg.anchor_view_multiplier - 1))
                ).bool(),
                torch.from_numpy(
                    np.repeat(applicable_sequence_indices, self.cfg.num_nearest_neighbors)
                ).bool()
            ])

            total_applicable_trajectories = applicable_trajectory_indices.sum().item()
            if total_applicable_trajectories > self.cfg.num_nearest_neighbors_inference:
                # transform applicable trajectories from binary to indices
                applicable_trajectory_indices = torch.arange(self.nearest_anchor_weight.shape[1], dtype=torch.long)[
                    applicable_trajectory_indices]
                # for each point, only use the num_nearest_neighbors_inference nearest neighbors (with highest weight)
                applicable_weights = self.nearest_anchor_weight[:, applicable_trajectory_indices]
                applicable_weights, applicable_indices = applicable_weights.topk(
                    self.cfg.num_nearest_neighbors_inference,
                    dim=1)
                applicable_trajectory_indices = applicable_trajectory_indices[applicable_indices.cpu()].to(
                    self.nearest_anchor_weight.device
                )

                anchor_indices = self.nearest_anchor_index.gather(1, applicable_trajectory_indices).view(
                    -1, self.cfg.num_nearest_neighbors_inference
                )
                anchor_offsets = self.nearest_anchor_offset.gather(1,
                                                                   applicable_trajectory_indices.unsqueeze(-1).expand(
                                                                       -1, -1, 3)).view(
                    -1, self.cfg.num_nearest_neighbors_inference, 3
                )
                anchor_weights = self.nearest_anchor_weight.gather(1, applicable_trajectory_indices).view(
                    -1, self.cfg.num_nearest_neighbors_inference
                )
                anchor_weights = anchor_weights / anchor_weights.sum(dim=1, keepdim=True)
            else:
                anchor_indices = self.nearest_anchor_index[:, applicable_trajectory_indices]
                anchor_offsets = self.nearest_anchor_offset[:, applicable_trajectory_indices]
                anchor_weights = self.nearest_anchor_weight[:, applicable_trajectory_indices]
                anchor_weights = anchor_weights / anchor_weights.sum(dim=1, keepdim=True)

            step_indices = torch.tensor(
                [(timestep + t) % self.trajectory_length for i, timestep in enumerate(self.timesteps) for _ in
                 range(self.num_trajectories[i])], dtype=torch.long
            )

            null_step_indices = torch.tensor(
                [timestep for i, timestep in enumerate(self.timesteps) for _ in range(self.num_trajectories[i])],
                dtype=torch.long
            )

            full_indices = torch.arange(self.anchor_trajectories.shape[1], dtype=torch.long)

            anchor_displacements = points_3d_world[step_indices, full_indices] - \
                                   points_3d_world[null_step_indices, full_indices]

            if self.cfg.inference_mode in ["displacement+rigid", "displacement"]:
                point_displacement = anchor_displacements[anchor_indices] * anchor_weights[..., None]
                point_displacement = point_displacement.sum(dim=1)

                from .utils import estimate_rigid_transform
                changes = estimate_rigid_transform(anchor_displacements, anchor_indices,
                                                   anchor_weights, anchor_offsets, point_offsets=point_displacement)
                if self.cfg.inference_mode == "displacement":
                    disp = point_displacement
                else:
                    disp = changes["displacement"] + point_displacement
            elif self.cfg.inference_mode == "rigid":
                from .utils import estimate_rigid_transform
                changes = estimate_rigid_transform(anchor_displacements, anchor_indices,
                                                   anchor_weights, anchor_offsets)
                disp = changes["displacement"]
            else:
                raise ValueError(f"Unknown inference mode: {self.cfg.inference_mode}")
            if self.movement_mask is not None:
                displacement[self.movement_mask, i] = disp
                rotation[self.movement_mask, i] = changes["rotation"]
                scaling[self.movement_mask, i] = changes["scale"]
            else:
                displacement[:, i] = disp
                rotation[:, i] = changes["rotation"]
                scaling[:, i] = changes["scale"]
            i += 1

        self.displacement = displacement
        self.rotation = rotation
        self.scaling = scaling

    def update(self, t, **kwargs):
        if hasattr(self, "displacement_gt") and self.displacement_gt is not None:
            trajectory_frame_times = torch.linspace(0.0, 1.0, self.trajectory_length)
            step_check = [math.isclose(t, frame_time, abs_tol=1e-3) for frame_time in trajectory_frame_times]
            assert any(step_check), "timestep not in saved trajectory"
            step = step_check.index(True)
            out = {
                "displacement": self.displacement_gt[step]
            }
            if self.cfg.update_rotation:
                out["rotation"] = self.rotation_gt[step]
            if self.cfg.update_scale:
                out["scale"] = self.scaling_gt[step]
            return out

        if not hasattr(self, "displacement") or self.displacement is None:
            out = {
                "displacement": torch.zeros_like(self._xyz),
            }
            if self.cfg.update_rotation:
                out["rotation"] = torch.zeros_like(self._rotation)
                out["rotation"][..., 0] = 1
            if self.cfg.update_scale:
                out["scale"] = torch.ones_like(self._scaling)
            return out

        trajectories_per_timestep = [
            sum([1 if 0 <= timestep + t < self.trajectory_length else 0 for timestep in self.timesteps])
            for t in
            range(- max(self.timesteps), self.trajectory_length - min(self.timesteps))]

        trajectories_in_vide_per_start_timestep = [sum(trajectories_per_timestep[i:self.trajectory_length + i]) for i in
                                                   range(
                                                       len(trajectories_per_timestep) - self.trajectory_length + 1)]
        max_trajectories_for_full_video, amount = max(
            zip(range(len(trajectories_in_vide_per_start_timestep) - 1, -1, -1),
                trajectories_in_vide_per_start_timestep[::-1]), key=lambda x: x[1])

        self.zero_timestep = max(self.timesteps) - max_trajectories_for_full_video

        displacement = self.displacement[:,
                       max_trajectories_for_full_video:max_trajectories_for_full_video + self.trajectory_length]
        rotation = self.rotation[:,
                   max_trajectories_for_full_video:max_trajectories_for_full_video + self.trajectory_length]
        scaling = self.scaling[:,
                  max_trajectories_for_full_video:max_trajectories_for_full_video + self.trajectory_length]

        trajectory_frame_times = torch.linspace(0.0, 1.0, self.trajectory_length)

        # check if t is at one of the sampled timesteps
        step_check = [math.isclose(t, frame_time, abs_tol=1e-3) for frame_time in trajectory_frame_times]
        if any(step_check):
            step = step_check.index(True)
            out = {
                "displacement": displacement[:, step],
            }
            if self.cfg.update_rotation:
                out["rotation"] = rotation[:, step]
            if self.cfg.update_scale:
                out["scale"] = scaling[:, step]
            return out
        elif t > 0 and t < 1:
            # perform linear interpolation between the nearest steps
            step = math.floor(t * (self.trajectory_length - 1))
            index_lower = step
            index_upper = step + 1
            weight_lower = t - index_lower / (self.trajectory_length - 1)
            weight_upper = index_upper / (self.trajectory_length - 1) - t

            weight_sum = weight_lower + weight_upper
            weight_lower = weight_upper / weight_sum
            weight_upper = (weight_sum - weight_upper) / weight_sum

            interpolated_displacement = weight_lower * displacement[:, index_lower] + \
                                        weight_upper * displacement[:, index_upper]
            interpolated_rotation = weight_lower * rotation[:, index_lower] + \
                                    weight_upper * rotation[:, index_upper]
            interpolated_scaling = weight_lower * scaling[:, index_lower] + \
                                   weight_upper * scaling[:, index_upper]

            out = {
                "displacement": interpolated_displacement,
            }
            if self.cfg.update_rotation:
                out["rotation"] = interpolated_rotation
            if self.cfg.update_scale:
                out["scale"] = interpolated_scaling
            return out
        else:
            def polynomial_extrapolate(tensor, t, degree=2):
                T = tensor.shape[0]

                x = torch.linspace(0, 1, T, dtype=torch.float32, device=tensor.device)
                X_vander = torch.vander(x, N=degree + 1, increasing=True)

                y = tensor.unsqueeze(1)
                # Solve the least squares problem
                coeffs = torch.linalg.lstsq(X_vander, y).solution
                # Predict the next point
                next_timestep = torch.tensor([t ** d for d in range(degree + 1)], dtype=torch.float32,
                                             device=tensor.device) @ coeffs
                return next_timestep

            # Extrapolate the displacement, rotation and scaling
            fun = lambda x: polynomial_extrapolate(x, t=t)
            out = {
                "displacement": torch.vmap(torch.vmap(fun, in_dims=1))(displacement).squeeze(-1),
            }
            if self.cfg.update_rotation:
                out["rotation"] = torch.vmap(torch.vmap(fun, in_dims=1))(rotation).squeeze(-1)
            if self.cfg.update_scale:
                out["scale"] = torch.vmap(torch.vmap(fun, in_dims=1))(scaling).squeeze(-1)
            return out

    def to(self, device="cpu"):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.max_sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1,
                                                                                                  2).contiguous().requires_grad_(
            False)

        if self.max_sh_degree > 0:
            self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1,
                                                                                                           2).contiguous().requires_grad_(
                False)

        else:
            self._features_rest = torch.tensor(features_dc, dtype=torch.float, device="cuda")[:, :, 1:].transpose(1,
                                                                                                                  2).contiguous().requires_grad_(
                False)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False)
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False)
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False)
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def save(self, path, timesteps=8):
        timesteps = torch.linspace(0, 1, timesteps)
        save = {
            "displacement": [],
        }
        if self.cfg.update_rotation:
            save["rotation"] = []
        if self.cfg.update_scale:
            save["scaling"] = []
        for t in timesteps:
            out = self.update(t)
            for k in save.keys():
                save[k].append(out[k].detach().cpu())

        for k in save.keys():
            save[k] = torch.stack(save[k], dim=0)
        torch.save(save, path)
        dprint("Saved deformations to {}".format(path))

    def load(self, path):
        data = torch.load(path)
        self.displacement_gt = data["displacement"].to(self.device)
        self.rotation_gt = data["rotation"].to(self.device) if "rotation" in data else None
        self.scaling_gt = data["scaling"].to(self.device) if "scaling" in data else None
        dprint("Loaded deformations from {}".format(path))
