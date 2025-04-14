from threestudio.utils.typing import *
from dataclasses import dataclass
from threestudio.utils.base import Updateable
from torch.utils.data import IterableDataset, DataLoader, Dataset
import pytorch_lightning as pl
from ..utils import sample_once_viewpoints_from_anchor_point
import numpy as np
import torch
import torch.nn.functional as F
import math
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)


@dataclass
class MultiviewSamplingOnceConfig:
    height: Any = 320
    width: Any = 512
    eval_height: Any = 320
    eval_width: Any = 512
    test_height: Any = 320
    test_width: Any = 512
    num_frames: int = 8
    eval_frames: int = 16
    test_extension: float = 1.0

    azimuth_range: Tuple[float, float] = (-90, 90)
    elevation_range: Tuple[float, float] = (-10, 10)
    distance_range: Tuple[float, float] = (-0.1, 0.1)
    eval_radius: float = 0.1
    std_azimuth: float = 0.1
    std_elevation: float = 0.2
    std_distance: float = 0.05
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    fovx_range: Optional[Tuple[float, float]] = None
    eval_fovy_deg: float = 70.0
    eval_fovx_deg: Optional[float] = None

    sample_rand_frames: Optional[str] = None
    num_test_loop_factor: int = 1

    camera_up: List[float] = (0., 0., 1.)
    camera_lookat: List[float] = (0., 0., 0.)
    anchor_view: List[float] = (1., 0., 0.)
    test_view: Optional[List[float]] = None

    num_views_per_direction: int = 10
    num_optimization_steps_before_resampling: int = 5000
    random_views: bool = False
    
    fixed_view: bool = False # If True, render test/val videos from a fixed view


class MultiviewSamplingOnceSampler(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        self.cfg: MultiviewSamplingOnceConfig = cfg
        self.fovx_range = self.cfg.fovy_range if self.cfg.fovx_range is None else self.cfg.fovx_range
        self.view_library = []
        self.directions_unit_focal = get_ray_directions(H=self.cfg.height, W=self.cfg.width, focal=1.0)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def __iter__(self):
        overall_view_index = 0
        while True:
            current_start_overall_view_index = overall_view_index

            yield {
                "index": 0,
                "generate_new": True,
                "overall_view_index": overall_view_index,
                "clear_previous": True,
                "reference_view": None,
                "view": self.cfg.anchor_view,
            }

            overall_view_index += 1

            views = sample_once_viewpoints_from_anchor_point(
                anchor_point=self.cfg.anchor_view,
                azimuth_range=self.cfg.azimuth_range,
                elevation_range=self.cfg.elevation_range,
                distance_range=self.cfg.distance_range,
                num_samples_per_direction=self.cfg.num_views_per_direction,
                std_azimuth=self.cfg.std_azimuth,
                std_elevation=self.cfg.std_elevation,
                std_distance=self.cfg.std_distance,
                lookat_center=self.cfg.camera_lookat
            )

            if self.cfg.random_views:
                view_permutation = np.random.permutation(views.shape[0])
                view_permutation_1 = np.stack([np.random.permutation(views.shape[1]) for _ in range(views.shape[0])],
                                              axis=0)

            for i in range(2):
                for j in range(self.cfg.num_views_per_direction):
                    yield {
                        "index": i * self.cfg.num_views_per_direction + j + 1,
                        "generate_new": True,
                        "overall_view_index": overall_view_index,
                        "clear_previous": False,
                        "reference_view": current_start_overall_view_index + i * self.cfg.num_views_per_direction + j if j > 0 else current_start_overall_view_index,
                        "view": views[i, j] if not self.cfg.random_views else views[
                            view_permutation[i], view_permutation_1[i, j]]
                    }
                    overall_view_index += 1

            gen = np.random.Generator(np.random.PCG64())

            for i in range(self.cfg.num_optimization_steps_before_resampling):
                random_index = gen.integers(0, self.cfg.num_views_per_direction * 2 + 1)

                yield {
                    "index": random_index,
                    "generate_new": False,
                    "overall_view_index": current_start_overall_view_index + random_index,
                    "clear_previous": False,
                    "reference_view": None,
                    "view": self.cfg.anchor_view if random_index == 0 else views[
                        (random_index - 1) // self.cfg.num_views_per_direction, (
                            random_index - 1) % self.cfg.num_views_per_direction]
                }

            self.view_library = []

    def collate(self, batch) -> Dict[str, Any]:
        batch_size = 1
        if batch["generate_new"]:
            camera_positions = torch.tensor(batch["view"], dtype=torch.float32).repeat(batch_size, 1)
            up = torch.tensor(self.cfg.camera_up).repeat(batch_size, 1)
            center = torch.tensor(self.cfg.camera_lookat).repeat(batch_size, 1)

            fovx_deg = (
                torch.rand(batch_size) * (self.cfg.fovx_range[1] - self.cfg.fovx_range[0]) + self.cfg.fovx_range[
                0]).repeat(batch_size)
            fovy_deg = (
                torch.rand(batch_size) * (self.cfg.fovy_range[1] - self.cfg.fovy_range[0]) + self.cfg.fovy_range[
                0]).repeat(batch_size)
            fovy = fovy_deg * math.pi / 180
            fovx = fovx_deg * math.pi / 180

            lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0

            # get directions by dividing directions_unit_focal by focal length
            focal_length_x: Float[Tensor, "B"] = (
                0.5 * self.cfg.height / torch.tan(0.5 * fovx)
            )
            focal_length_y: Float[Tensor, "B"] = (
                0.5 * self.cfg.height / torch.tan(0.5 * fovy)
            )
            directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                                   None, :, :, :
                                                   ].repeat(batch_size, 1, 1, 1)

            directions[:, :, :, 0] = directions[:, :, :, 0] / focal_length_x[:, None, None,
                                                              None]  # Apply x focal length
            directions[:, :, :, 1] = directions[:, :, :, 1] / focal_length_y[:, None, None,
                                                              None]  # Apply y focal length

            # Importance note: the returned rays_d MUST be normalized!
            rays_o, rays_d = get_rays(
                directions, c2w, keepdim=True, normalize=True
            )
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, self.cfg.width / self.cfg.height, 0.01, 100.0
            )  # FIXME: hard-coded near and far
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

            if self.cfg.sample_rand_frames == "t0":
                rand_gen = torch.Generator()
                rand_gen.seed()
                t0 = torch.rand(1, generator=rand_gen).item() / self.cfg.num_frames
                # t0 = torch.FloatTensor(1).uniform_(0, 1 / self.cfg.num_frames).item()
                frame_times = torch.linspace(
                    t0, t0 + (self.cfg.num_frames - 1) / self.cfg.num_frames, self.cfg.num_frames
                )
            elif self.cfg.sample_rand_frames == "all":
                rand_gen = torch.Generator()
                rand_gen.seed()
                frame_times = torch.linspace(0.0, 1.0, self.cfg.num_frames)
                perturb = torch.randn(self.cfg.num_frames, generator=rand_gen) * 0.02
                frame_times = frame_times + perturb
                frame_times = torch.clamp(frame_times, 0.0, 1.0)
            else:
                frame_times = torch.linspace(0.0, 1.0, self.cfg.num_frames)

            batch_ = {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "mvp_mtx": mvp_mtx,
                "camera_positions": camera_positions,
                "c2w": c2w,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "fovx": fovx,
                "fovy": fovy,
                "proj_mtx": proj_mtx,
                "frame_times": frame_times,
                "frame_times_video": frame_times,
                "is_video": True,
                "train_dynamic_camera": False,
                "view_index": batch["overall_view_index"],
                "reference_view": batch["reference_view"],
            }
            self.view_library.append(batch_)

        return self.view_library[batch["index"]] | {"clear_previous": batch["clear_previous"]}


class AnchorViewDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewSamplingOnceConfig = cfg
        self.split = split

        # --- Determine Height/Width based on split ---
        if split == "val":
            self.n_views = 1 # Render one video for validation
            height = self.cfg.eval_height
            width = self.cfg.eval_width
            fovx_deg = torch.tensor(self.cfg.eval_fovx_deg if self.cfg.eval_fovx_deg is not None else 60.0)
            fovy_deg = torch.tensor(self.cfg.eval_fovy_deg)
            # eval_frames are used for val batch size
            batch_size = self.cfg.eval_frames

        elif split == "test":
            self.n_views = 1 # Render one video for testing
            height = self.cfg.test_height
            width = self.cfg.test_width
            # Use eval FOV for test view as well unless specified otherwise
            fovx_deg = torch.tensor(self.cfg.eval_fovx_deg if self.cfg.eval_fovx_deg is not None else 60.0)
            fovy_deg = torch.tensor(self.cfg.eval_fovy_deg if self.cfg.eval_fovy_deg is not None else 40.0)
            # test_extension scales the number of frames for test
            batch_size = int(self.cfg.eval_frames * self.cfg.test_extension)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.directions_unit_focal = get_ray_directions(H=height, W=width, focal=1.0)
        lookat_point = np.array(self.cfg.camera_lookat)

        # +++ CONDITIONAL CAMERA POSE GENERATION +++
        if self.cfg.fixed_view:
            # --- FIXED VIEW LOGIC ---
            print(f"[INFO] Generating fixed view camera poses for {split} split.")
            anchor_point_np = np.array(self.cfg.anchor_view)
            single_camera_position = torch.tensor(anchor_point_np, dtype=torch.float32)
            # Repeat the anchor camera position for all frames
            camera_positions = single_camera_position.unsqueeze(0).repeat(batch_size, 1)
            train_dynamic_camera = False # Camera is static

            # Calculate camera parameters for the fixed view
            up = torch.tensor(self.cfg.camera_up).repeat(batch_size, 1)
            center = torch.tensor(self.cfg.camera_lookat).repeat(batch_size, 1)
            fovy = (fovy_deg * math.pi / 180).repeat(batch_size)
            fovx = (fovx_deg * math.pi / 180).repeat(batch_size)

            lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
            right: Float[Tensor, "B 3"] = F.normalize(torch.linalg.cross(lookat, up, dim=-1), dim=-1)
            up_final = F.normalize(torch.linalg.cross(right, lookat, dim=-1), dim=-1) # Renamed to avoid clash
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [torch.stack([right, up_final, -lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0

            focal_length_x: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovx)
            focal_length_y: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)

            directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                                None, :, :, :
                                                ].repeat(batch_size, 1, 1, 1)
            directions[:, :, :, 0] = directions[:, :, :, 0] / focal_length_x[:, None, None]
            directions[:, :, :, 1] = directions[:, :, :, 1] / focal_length_y[:, None, None]

            rays_o, rays_d = get_rays(
                directions, c2w, keepdim=True, normalize=True
            )
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, width / height, 0.01, 100.0
            )
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

            # Frame times for fixed view
            if self.split == "val":
                frame_times = torch.linspace(0.0, 1.0, batch_size)
            else: # test split
                frame_times = torch.linspace(0.0, self.cfg.test_extension, batch_size)
            # --- END FIXED VIEW LOGIC ---

        else:
            # --- ORIGINAL ORBITING + CIRCULAR VIEW LOGIC ---
            print(f"[INFO] Generating orbiting camera poses for {split} split.")

            # Determine anchor point based on split
            if self.cfg.test_view is not None and split=='test':
                 anchor_point = np.array(self.cfg.test_view) - lookat_point
            else:
                 anchor_point = np.array(self.cfg.anchor_view) - lookat_point

            r = np.linalg.norm(anchor_point)
            anchor_phi = np.arctan2(anchor_point[1], anchor_point[0])
            anchor_elevation = np.arccos(anchor_point[2] / r)

            # == Calculate Orbiting Poses ==
            orbit_batch_size = batch_size # Use initial batch_size for orbit
            points_orbit = np.linspace(0, 2 * np.pi, orbit_batch_size, endpoint=False)
            azimuth_change_orbit = np.cos(points_orbit) * self.cfg.eval_radius
            elevation_change_orbit = np.sin(points_orbit) * self.cfg.eval_radius
            azimuth_orbit = anchor_phi + azimuth_change_orbit
            elevation_orbit = anchor_elevation + elevation_change_orbit
            pos_x_orbit = r * np.sin(elevation_orbit) * np.cos(azimuth_orbit)
            pos_y_orbit = r * np.sin(elevation_orbit) * np.sin(azimuth_orbit)
            pos_z_orbit = r * np.cos(elevation_orbit)
            camera_positions_orbit = np.stack([pos_x_orbit, pos_y_orbit, pos_z_orbit], axis=1) + lookat_point[None]
            camera_positions_orbit = torch.tensor(camera_positions_orbit, dtype=torch.float32)

            # == Calculate Circular Poses ==
            circ_batch_size = orbit_batch_size * 4 # As originally calculated
            points_circ = np.linspace(0, 2 * np.pi, circ_batch_size, endpoint=False)
            azimuth_change_circ = points_circ
            elevation_change_circ = np.zeros_like(azimuth_change_circ)
            azimuth_circ = anchor_phi + azimuth_change_circ
            elevation_circ = anchor_elevation + elevation_change_circ
            pos_x_circ = r * np.sin(elevation_circ) * np.cos(azimuth_circ)
            pos_y_circ = r * np.sin(elevation_circ) * np.sin(azimuth_circ)
            pos_z_circ = r * np.cos(elevation_circ)
            camera_positions_circ = np.stack([pos_x_circ, pos_y_circ, pos_z_circ], axis=1) + lookat_point[None]
            camera_positions_circ = torch.tensor(camera_positions_circ, dtype=torch.float32)

            # == Combine Poses ==
            camera_positions = torch.cat([camera_positions_orbit, camera_positions_circ], dim=0)
            combined_batch_size = camera_positions.shape[0] # Total number of poses

            # == Calculate Camera Parameters for Combined Poses ==
            up = torch.tensor(self.cfg.camera_up).repeat(combined_batch_size, 1)
            center = torch.tensor(self.cfg.camera_lookat).repeat(combined_batch_size, 1)
            fovy_combined = (fovy_deg * math.pi / 180).repeat(combined_batch_size)
            fovx_combined = (fovx_deg * math.pi / 180).repeat(combined_batch_size)

            lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
            right: Float[Tensor, "B 3"] = F.normalize(torch.linalg.cross(lookat, up, dim=-1), dim=-1)
            up_final = F.normalize(torch.linalg.cross(right, lookat, dim=-1), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [torch.stack([right, up_final, -lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0

            focal_length_x: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovx_combined)
            focal_length_y: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy_combined)

            directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                                    None, :, :, :
                                                    ].repeat(combined_batch_size, 1, 1, 1)
            directions[:, :, :, 0] = directions[:, :, :, 0] / focal_length_x[:, None, None]
            directions[:, :, :, 1] = directions[:, :, :, 1] / focal_length_y[:, None, None]

            rays_o, rays_d = get_rays(
                directions, c2w, keepdim=True, normalize=True
            )
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy_combined, width / height, 0.01, 100.0
            )
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

            # == Frame times for combined orbit + circular ==
            if self.split == "val":
                frame_times_orbit = torch.linspace(0.0, 1.0, orbit_batch_size)
            else: # test split
                frame_times_orbit = torch.linspace(0.0, self.cfg.test_extension, orbit_batch_size)
            frame_times_circ = frame_times_orbit.repeat(4) # Repeat for circular views
            frame_times = torch.cat([frame_times_orbit, frame_times_circ], dim=0)

            train_dynamic_camera = True # Camera is dynamic
            # --- END ORIGINAL ORBITING + CIRCULAR VIEW LOGIC ---


        # --- Create the final batch dictionary (common to both modes) ---
        self.batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "height": height,
            "width": width,
            "fovx": fovx if self.cfg.fixed_view else fovx_combined, # Use correct fov
            "fovy": fovy if self.cfg.fixed_view else fovy_combined, # Use correct fov
            "proj_mtx": proj_mtx,
            "frame_times": frame_times,
            "frame_times_video": frame_times, # Use the same times for video saving
            "is_video": True,
            "train_dynamic_camera": train_dynamic_camera,
        }

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return self.batch

    def collate(self, batch):
        # The dataloader gives us a list containing one item (self.batch)
        # So we just return that item.
        if isinstance(batch, list):
             return batch[0]
        return batch # Should already be the dictionary

@register("4dgs-multi-view-sample-once-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: MultiviewSamplingOnceConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewSamplingOnceConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiviewSamplingOnceSampler(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = AnchorViewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = AnchorViewDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
