from dataclasses import dataclass, field

import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
from threestudio.utils.misc import get_device

import time
from ..utils import set_debug, dprint, FlowBackProjection, build_intrinsics


@threestudio.register("gauss-trajectory-4d-system")
class GaussTo4D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        guidance_type: str = ""
        guidance: dict = field(default_factory=dict)

        prompt_processor_type: str = ""
        prompt_processor: dict = field(default_factory=dict)

        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        debug: bool = False
        last_step_rigid: bool = False

    cfg: Config

    def configure(self) -> None:
        set_debug(self.cfg.debug)
        self.automatic_optimization = False
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(
            self.cfg.guidance
        )
        self.prompt_utils = [self.cfg.prompt_processor.get("prompt", "")]

        self.actual_step = -1
        self.flow_back_projection = None

    def configure_optimizers(self):
        return []

    def forward(self, batch: Dict[str, Any], testing=False) -> Dict[str, Any]:
        if self.flow_back_projection is None:
            intrinsics = build_intrinsics(batch["fovx"][0], batch["fovy"][0], batch["width"], batch["height"])

            self.flow_back_projection = FlowBackProjection(
                camera_intrinsics=intrinsics,
                device=get_device()
            )

        render_outs = []
        batch["frame_times"] = batch["frame_times"].flatten()
        for frame_idx, frame_time in enumerate(batch["frame_times"].tolist()):
            if batch["train_dynamic_camera"]:
                batch_frame = {}
                for k_frame, v_frame in batch.items():
                    if isinstance(v_frame, torch.Tensor):
                        if v_frame.shape[0] == batch["frame_times"].shape[0]:
                            v_frame_up = v_frame[[frame_idx]].clone()
                        else:
                            v_frame_up = v_frame.clone()
                    else:
                        v_frame_up = v_frame
                    batch_frame[k_frame] = v_frame_up
                batch_frame["time"] = frame_time
                batch_frame["time_step"] = frame_idx
                render_out = self.renderer.batch_forward(batch_frame)
            else:
                batch["time"] = frame_time
                batch["time_step"] = frame_idx
                render_out = self.renderer.batch_forward(batch)

            render_out_ = {}
            render_out_["comp_rgb"] = render_out["comp_rgb"].detach()
            render_out_["rendered_depth"] = render_out["rendered_depth"].detach()
            if testing:
                render_out_["comp_rgb"] = render_out_["comp_rgb"].cpu().detach()
                render_out_["rendered_depth"] = render_out_["rendered_depth"].cpu().detach()
                render_out_["positions"] = render_out["updated_vars"]["xyz"][None].detach()
                render_out_["rotations"] = render_out["updated_vars"]["rotation"][None].detach()
                render_out_["scalings"] = render_out["updated_vars"]["scaling"][None].detach()
                render_out_["updated_pos"] = render_out["changes"]["displacement"][None].detach()
                if "rotation" in render_out["changes"]:
                    render_out_["updated_rot"] = render_out["changes"]["rotation"][None].detach()
                if "scale" in render_out["changes"]:
                    render_out_["updated_scale"] = render_out["changes"]["scale"][None].detach()
            render_out = render_out_
            render_outs.append(render_out)

        out = {}
        for k in render_out:
            out[k] = torch.cat(
                [render_out_i[k] if not isinstance(render_out_i[k], list) else render_out_i[k][0][None] for
                 render_out_i in render_outs]
            )

        # add intrinsics and extrinsics to out
        out["intrinsics"] = build_intrinsics(batch["fovx"][0], batch["fovy"][0], batch["width"], batch["height"])
        out["extrinsics"] = batch["c2w"]

        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):

        self.actual_step += 1

        if hasattr(self.guidance, "update_t_w"):
            self.guidance.update_t_w(self.actual_step, self.trainer.max_steps)

        if self.actual_step == self.trainer.max_steps - 1 and self.cfg.last_step_rigid:
            self.geometry.cfg.inference_mode = "rigid"

        prompt_utils = self.prompt_utils
        start = time.time()
        out = self(batch)
        dprint(f"0) Forward time: {time.time() - start:.4f}")
        batch["num_frames"] = self.cfg.geometry["num_frames"]

        start = time.time()
        guidance_inp = out["comp_rgb"]
        with torch.no_grad():
            self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False, zero_timestep=self.geometry.zero_timestep
            )
        dprint(f"1) Guidance time: {time.time() - start:.4f}")

        guidance_video = self.guidance.diffusion_output.permute(0, 2, 1, 3, 4)

        start = time.time()
        # find timestep which best aligns with no deformation timestep
        static_view = out["comp_rgb"][self.geometry.zero_timestep].permute(2, 0, 1)
        per_frame_diffs = torch.abs(guidance_video[0] - static_view).mean(dim=3).mean(dim=2).mean(dim=1)
        timestep = torch.argmin(per_frame_diffs).item()
        dprint(f"2) Find timestep time: {time.time() - start:.4f}, timestep: {timestep}")

        start = time.time()
        trajectories, depth_confidence = self.flow_back_projection.back_project(
            guidance_video, out["rendered_depth"][self.geometry.zero_timestep, 0], timestep=timestep
        )
        dprint(f"3) Back projection time: {time.time() - start:.4f}")

        start = time.time()
        self.geometry.register_trajectories(trajectories, out["extrinsics"][0], depth_confidence, timestep)
        dprint(f"4) Register trajectories time: {time.time() - start:.4f}")

        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += 1

        return {"loss": torch.tensor(0.0)}

    def save_video(self, batch_video, step):
        if isinstance(batch_video, dict):
            out_video = batch_video["comp_rgb"]
        else:
            out_video = batch_video

        for index in range(out_video.shape[0]):
            self.save_image_grid(
                f"it{step}/{index}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out_video[index],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="validation_step",
                step=step,
            )
        self.save_img_sequence(
            f"it{step}",
            f"it{step}",
            "(\d+)\.png",
            save_format="gif",
            fps=16,
            name=f"test_static",
        )

    def validation_step(self, batch, batch_idx):
        start = time.time()
        batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
        batch_video["frame_times"] = batch["frame_times_video"]
        out_video = self(batch_video, testing=True)
        self.save_video(out_video, f"{self.true_global_step}-eval")
        dprint(f"A) Evaluation time: {time.time() - start:.4f}")

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # free VRAM by deleting diffusion guidance etc.
        self.guidance = None
        self.flow_back_projection = 1

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        with torch.no_grad():
            start = time.time()
            batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_video["frame_times"] = batch["frame_times_video"]
            out = self.forward(batch_video, testing=True)
        self.save_video(out, f"{self.true_global_step}-test")
        dprint(f"B) Test time video: {time.time() - start:.4f}")

        # Evaluation
        start = time.time()

        from ..utils import rigidity_loss, arap_loss, rotation_similarity_loss, longterm_isometry_loss, jsd_loss
        from ..evaluation import clip_scores

        # Create pseudo-geometry
        class GeometryObject:
            def __init__(self, init_xyz, num_nearest_neighbors=30):
                from ..geometry.utils import o3d_knn
                self.knn_indices, self.knn_relative_positions, self.knn_squared_dists = o3d_knn(init_xyz,
                                                                                                num_nearest_neighbors)
                print(f"KNN computation took {time.time() - start} seconds")
                self.knn_weights = torch.exp(-self.knn_squared_dists)
                self.knn_weights = self.knn_weights / torch.sum(self.knn_weights, dim=1, keepdim=True)

        geometry = GeometryObject(self.geometry.get_xyz)

        # Compute losses
        displacement = torch.abs(out["updated_pos"]).mean() / len(out["updated_pos"])
        if "updated_rot" in out:
            rotation = torch.abs(
                out["updated_rot"] - torch.tensor([1., 0., 0., 0.], device=out["updated_rot"].device)).mean() / len(
                out["updated_rot"])
        if "updated_scale" in out:
            scale = torch.abs(out["updated_scale"] - 1.).mean() / len(out["updated_scale"])
        rigidity = rigidity_loss(out["positions"], out["rotations"], geometry)
        momentum = torch.abs(out["updated_pos"][1:] - out["updated_pos"][:-1]).mean() / len(out["updated_pos"] - 1)
        arap = arap_loss(out["positions"], geometry)
        jsd = jsd_loss(out["positions"])
        isometry = longterm_isometry_loss(out["positions"], geometry)
        rotation_sim = rotation_similarity_loss(out["rotations"], geometry)

        # print results in nice table
        print("-------------------RESULTS-------------------")
        print(f"{'Displacement':<20}\t{displacement.item():.8f}")
        if "updated_rot" in out:
            print(f"{'Rotation':<20}\t{rotation.item():.8f}")
        if "updated_scale" in out:
            print(f"{'Scale':<20}\t{scale.item():.8f}")
        print(f"{'Rigidity':<20}\t{rigidity.item():.8f}")
        print(f"{'Momentum':<20}\t{momentum.item():.8f}")
        print(f"{'ARAP':<20}\t{arap.item():.8f}")
        print(f"{'JSD':<20}\t{jsd.item():.8f}")
        print(f"{'Isometry':<20}\t{isometry.item():.8f}")
        print(f"{'Rotation similarity':<20}\t{rotation_sim.item():.8f}")
        print("-------------------RESULTS-------------------")

        # Compute CLIP scores
        print(out["comp_rgb"].shape)
        naive, advanced = clip_scores(out["comp_rgb"].permute(0, 3, 1, 2), self.prompt_utils)

        print(f"Naive CLIP score: {naive:.8f}")
        print(f"Advanced CLIP score: {advanced:.8f}")
        print("-------------------RESULTS-------------------")

        dprint(f"C) Test time evaluation: {time.time() - start}")

    def on_test_epoch_end(self):
        pass
