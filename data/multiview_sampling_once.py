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

        batch_size = self.cfg.eval_frames
        if split == "test":
            batch_size = int(batch_size * self.cfg.test_extension)

        if split == "val":
            self.n_views = 1
        else:
            self.n_views = 1

        self.directions_unit_focal = get_ray_directions(H=self.cfg.eval_height, W=self.cfg.eval_width,
                                                        focal=1.0) if split == "val" else get_ray_directions(
            H=self.cfg.test_height, W=self.cfg.test_width, focal=1.0)

        lookat_point = np.array(self.cfg.camera_lookat)
        if self.cfg.test_view is not None:
            anchor_point = np.array(self.cfg.test_view) - lookat_point
        else:
            anchor_point = np.array(self.cfg.anchor_view) - lookat_point

        # Normalize radius if not provided or if radius is different from the calculated one.
        r = np.linalg.norm(anchor_point)

        # Determine azimuth and elevation of the anchor point
        anchor_phi = np.arctan2(anchor_point[1], anchor_point[0])
        anchor_elevation = np.arccos(anchor_point[2] / r)

        # Generate azimuth and elevation angles in a circular manner
        points = np.linspace(0, 2 * np.pi, batch_size)
        azimuth_change = np.cos(points) * self.cfg.eval_radius
        elevation_change = np.sin(points) * self.cfg.eval_radius

        azimuth = anchor_phi + azimuth_change
        elevation = anchor_elevation + elevation_change

        # Calculate the camera positions
        pos_x = r * np.sin(elevation) * np.cos(azimuth)
        pos_y = r * np.sin(elevation) * np.sin(azimuth)
        pos_z = r * np.cos(elevation)

        # Translate positions to be relative to the lookat point
        camera_positions = np.stack([pos_x, pos_y, pos_z], axis=1) + lookat_point[None]

        # Convert to torch tensor
        camera_positions = torch.tensor(camera_positions, dtype=torch.float32)

        up = torch.tensor(self.cfg.camera_up).repeat(batch_size, 1)
        center = torch.tensor(self.cfg.camera_lookat).repeat(batch_size, 1)

        fovx_deg = torch.tensor(self.cfg.eval_fovx_deg).repeat(batch_size)
        fovy_deg = torch.tensor(self.cfg.eval_fovy_deg).repeat(batch_size)
        fovy = fovy_deg * math.pi / 180
        fovx = fovx_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.linalg.cross(lookat, up), dim=-1)
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

        directions[:, :, :, 0] = directions[:, :, :, 0] / focal_length_x[:, None, None]  # Apply x focal length
        directions[:, :, :, 1] = directions[:, :, :, 1] / focal_length_y[:, None, None]  # Apply y focal length

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=True
        )
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.01, 100.0
        ) if split == "val" else get_projection_matrix(
            fovy, self.cfg.test_width / self.cfg.test_height, 0.01, 100.0
        )
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        if self.split == "val":
            frame_times = torch.linspace(0.0, 1.0, batch_size)
        else:
            frame_times = torch.linspace(0.0, self.cfg.test_extension, batch_size)

        # add circular views around object
        points = np.linspace(0, 2 * np.pi, batch_size * 4)
        # azimuth_change = np.sin(points)
        azimuth_change = points
        elevation_change = np.zeros_like(azimuth_change)

        azimuth = anchor_phi + azimuth_change
        elevation = anchor_elevation + elevation_change

        pos_x = r * np.sin(elevation) * np.cos(azimuth)
        pos_y = r * np.sin(elevation) * np.sin(azimuth)
        pos_z = r * np.cos(elevation)

        camera_positions_ = np.stack([pos_x, pos_y, pos_z], axis=1) + lookat_point[None]
        camera_positions_ = torch.tensor(camera_positions_, dtype=torch.float32)

        up = torch.tensor(self.cfg.camera_up).repeat(batch_size * 4, 1)
        center = torch.tensor(self.cfg.camera_lookat).repeat(batch_size * 4, 1)

        fovx_deg = torch.tensor(self.cfg.eval_fovx_deg).repeat(batch_size * 4)
        fovy_deg = torch.tensor(self.cfg.eval_fovy_deg).repeat(batch_size * 4)
        fovy_ = fovy_deg * math.pi / 180
        fovx_ = fovx_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions_, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions_[:, :, None]],
            dim=-1,
        )
        c2w_: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w_[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length_x: Float[Tensor, "B"] = (
            0.5 * self.cfg.height / torch.tan(0.5 * fovx_)
        )
        focal_length_y: Float[Tensor, "B"] = (
            0.5 * self.cfg.height / torch.tan(0.5 * fovy_)
        )
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                               None, :, :, :
                                               ].repeat(batch_size * 4, 1, 1, 1)

        directions[:, :, :, 0] = directions[:, :, :, 0] / focal_length_x[:, None, None]  # Apply x focal length
        directions[:, :, :, 1] = directions[:, :, :, 1] / focal_length_y[:, None, None]  # Apply y focal length

        # Importance note: the returned rays_d MUST be normalized!
        rays_o_, rays_d_ = get_rays(
            directions, c2w_, keepdim=True, normalize=True
        )
        proj_mtx_: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy_, self.cfg.eval_width / self.cfg.eval_height, 0.01, 100.0
        ) if split == "val" else get_projection_matrix(
            fovy_, self.cfg.test_width / self.cfg.test_height, 0.01, 100.0
        )
        mvp_mtx_: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w_, proj_mtx_)

        if self.split == "val":
            frame_times_ = torch.linspace(0.0, 1.0, batch_size).repeat(4)
        else:
            frame_times_ = torch.linspace(0.0, self.cfg.test_extension, batch_size).repeat(4)

        self.batch = {
            "rays_o": torch.cat([rays_o, rays_o_], dim=0),
            "rays_d": torch.cat([rays_d, rays_d_], dim=0),
            "mvp_mtx": torch.cat([mvp_mtx, mvp_mtx_], dim=0),
            "camera_positions": torch.cat([camera_positions, camera_positions_], dim=0),
            "c2w": torch.cat([c2w, c2w_], dim=0),
            "height": self.cfg.eval_height if split == "val" else self.cfg.test_height,
            "width": self.cfg.eval_width if split == "val" else self.cfg.test_width,
            "fovx": torch.cat([fovx, fovx_], dim=0),
            "fovy": torch.cat([fovy, fovy_], dim=0),
            "proj_mtx": torch.cat([proj_mtx, proj_mtx_], dim=0),
            "frame_times": torch.cat([frame_times, frame_times_], dim=0),
            "frame_times_video": torch.cat([frame_times, frame_times_], dim=0),
            "is_video": True,
            "train_dynamic_camera": True,
        }

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return self.batch

    def collate(self, batch):
        return self.batch


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
