from dataclasses import dataclass, field


from threestudio.utils.ops import get_cam_info_gaussian 

import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
from threestudio.utils.misc import get_device

import imageio
import os
import numpy as np
import cv2
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch.nn.functional as F


import time
from ..utils import set_debug, dprint, FlowBackProjection, build_intrinsics

from lpips import LPIPS
from ..utils.loss_utils import ssim

import torch.optim as optim


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
        save_debug_artifacts: bool = False

        use_optimization: bool = False
        optimizer: dict = field(default_factory=dict)
        loss: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        set_debug(self.cfg.debug)
        self.automatic_optimization = False
        # set up geometry, material, background, renderer
        super().configure()
        
        # ===================================================================
        # ============== CONDITIONAL SETUP BASED ON THE FLAG ================
        # ===================================================================
        
        if self.cfg.use_optimization:
            threestudio.info("****configuring system for OPTIMIZATION mode****")
            
            
            # 1. Initialize geometry for learnable motion
            #The number of keyframes == number of frames in the video
            num_frames = self.cfg.geometry["num_frames"]  
            self.geometry.initialize_learnable_motion(num_frames)
            
            #2. Load the target video for supervision
            video_path = self.cfg.guidance.anchor_viewpoint_sample_path
            if not os.path.exists(video_path) or not video_path:
                raise ValueError(f"Anchor viewpoint sample video path not found: {video_path}")
            
            threestudio.info(f"Loading anchor viewpoint sample video from {video_path}")
            reader = imageio.get_reader(video_path, 'ffmpeg')
            
            frames = []
            for i, frame in enumerate(reader):
                if i >= num_frames:
                    break
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frames.append(frame_tensor)
            
            self.target_video_frames = torch.stack(frames, dim=0).to(self.device).permute(0, 3, 1, 2) # To T,C,H,W
            
            threestudio.info(f"Loaded {self.target_video_frames.shape[0]} frames for optimization. Shape: {self.target_video_frames.shape}")
            
            #3. Set up losses
            
            self.lpips_loss = LPIPS(net='vgg').to(self.device)
            
        else:
            threestudio.info("***** Configuring system for INFERENCE-LIFTING mode. *****")
            
            self.guidance = threestudio.find(self.cfg.guidance_type)(
                self.cfg.guidance
            )
            self.prompt_utils = [self.cfg.prompt_processor.get("prompt", "")]

            self.flow_back_projection = None

        self.actual_step = -1
        
    def configure_optimizers(self):
        if self.cfg.use_optimization:
            threestudio.info("Configuring Adam optimizer for motion parameters.")
            optim_cfg = self.cfg.optimizer
            # optimizer_class = threestudio.find(optim_cfg.optimizer_type)
            optimizer = optim.Adam(self.geometry.get_trainable_motion_parameters(), **optim_cfg.args)
            return optimizer
        
        return []
    
    def _get_debug_save_dir(self, step):
        """Get the directory for saving debug artifacts for a specific step."""
        base_dir = os.path.join(self.get_save_dir(), "debugging_files")
        step_dir = os.path.join(base_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def _save_video_as_frames_and_gif(self, video_tensor, save_dir, prefix):
        """Save a video tensor as individual frames and a GIF."""
        dprint(f"Input video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        
        # Handle different input formats
        if video_tensor.dim() == 5:
            # Check if it's (B, C, T, H, W) format from LTX
            if video_tensor.shape[1] == 3 and video_tensor.shape[2] > 10:  # C=3 and T > 10 suggests (B, C, T, H, W)
                dprint(f"Detected (B, C, T, H, W) format, converting to (B, T, C, H, W)")
                video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
            # Now it should be (B, T, C, H, W)
            video_tensor = video_tensor[0]  # Remove batch dimension to get (T, C, H, W)
            dprint(f"After removing batch dimension: {video_tensor.shape}")
        
        if video_tensor.dim() == 4:
            # Check if dimensions look like (C, T, H, W) instead of (T, C, H, W)
            if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 10:  # C=3 and T > 10
                dprint(f"Detected (C, T, H, W) format, converting to (T, C, H, W)")
                video_tensor = video_tensor.permute(1, 0, 2, 3)
            # Now convert (T, C, H, W) to (T, H, W, C)
            video_tensor = video_tensor.permute(0, 2, 3, 1)
            dprint(f"After permuting to (T, H, W, C): {video_tensor.shape}")

        # ensure all pixel values are in [0, 1]
        if video_tensor.min() < 0.0:
            # Typical range for generative models is [-1, 1]; shift to [0, 1]
            dprint(f"Video tensor has negative values (min: {video_tensor.min():.3f}), normalizing to [0,1] range.")
            video_tensor = (video_tensor.clamp(-1.0, 1.0) + 1.0) * 0.5
        
        
        # Ensure values are in [0, 1] range
        if video_tensor.max() > 1.0:
            dprint(f"Video tensor has values > 1.0 (max: {video_tensor.max()}), normalizing...")
            video_tensor = video_tensor / 255.0
        
        # Save individual frames
        frames_dir = os.path.join(save_dir, f"{prefix}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        dprint(f"Created frames directory: {frames_dir}")
        
        frames = []
        for t in range(video_tensor.shape[0]):
            frame = cv2.normalize(
                (video_tensor[t].cpu().numpy()), None,
                alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            ).astype(np.uint8)
            # Ensure frame has correct shape (H, W, C) with C=3
            if frame.shape[-1] != 3:
                dprint(f"ERROR: Frame {t} has shape {frame.shape}, expected (H, W, 3)")
                continue
            frame_path = os.path.join(frames_dir, f"frame_{t:04d}.png")
            success = cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not success:
                dprint(f"Failed to save frame {t} to {frame_path}")
            frames.append(frame)
            if t == 0:
                dprint(f"First frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
        
        # Save as GIF
        gif_path = os.path.join(save_dir, f"{prefix}.gif")
        try:
            imageio.mimsave(gif_path, frames, fps=8)
            dprint(f"Saved GIF to {gif_path}")
        except Exception as e:
            dprint(f"Failed to save GIF: {e}")
            import traceback
            traceback.print_exc()
        
        return frames_dir, gif_path

    def _save_depth_maps(self, depth_maps, confidence_maps, save_dir):
        """Save depth maps and confidence maps as visualizations."""
        depth_dir = os.path.join(save_dir, "depth_maps")
        os.makedirs(depth_dir, exist_ok=True)
        
        # Lists to store frames for video creation
        depth_frames = []
        confidence_frames = []
        
        for t in range(depth_maps.shape[0]):
            # Normalize depth map for visualization
            depth_map = depth_maps[t].cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = (depth_map * 255).astype(np.uint8)
            
            # Save depth map
            depth_path = os.path.join(depth_dir, f"depth_{t:04d}.png")
            cv2.imwrite(depth_path, depth_map)
            depth_frames.append(depth_map)
            
            # Save confidence map
            conf_map = confidence_maps[t].cpu().numpy()
            conf_map = (conf_map * 255).astype(np.uint8)
            conf_path = os.path.join(depth_dir, f"confidence_{t:04d}.png")
            cv2.imwrite(conf_path, conf_map)
            confidence_frames.append(conf_map)
        
        # Save videos
        depth_video_path = os.path.join(depth_dir, "depth_video.gif")
        confidence_video_path = os.path.join(depth_dir, "confidence_video.gif")
        imageio.mimsave(depth_video_path, depth_frames, fps=8)
        imageio.mimsave(confidence_video_path, confidence_frames, fps=8)

    def _save_point_tracks(self, video, tracks, visibility, save_dir):
        """Save point tracks visualization."""
        tracks_dir = os.path.join(save_dir, "point_tracks")
        os.makedirs(tracks_dir, exist_ok=True)
        
        # Convert video to numpy for visualization
        video_np = (video[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        tracks_np = tracks[0].cpu().numpy()
        visibility_np = visibility[0].cpu().numpy()
        
        # List to store frames for video creation
        track_frames = []
        
        for t in range(video_np.shape[0]):
            frame = video_np[t].copy()
            
            # Draw visible points
            visible_points = tracks_np[t][visibility_np[t] > 0]
            for point in visible_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Save frame with tracks
            track_path = os.path.join(tracks_dir, f"tracks_{t:04d}.png")
            cv2.imwrite(track_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            track_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save video
        track_video_path = os.path.join(tracks_dir, "tracks_video.gif")
        imageio.mimsave(track_video_path, track_frames, fps=8)

    def _save_3d_trajectories(self, points_3d, save_dir):
        """Save 3D trajectories as a static 3D plot visualization."""
        traj_dir = os.path.join(save_dir, "3d_trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        dprint(f"Saving static 3D trajectories to {traj_dir}")

        points_np = points_3d.cpu().numpy() # Shape: (T, N, 3)

        # Pre-calculate axis limits across all frames for a stable view in the GIF
        x_lim = (points_np[..., 0].min(), points_np[..., 0].max())
        y_lim = (points_np[..., 1].min(), points_np[..., 1].max())
        z_lim = (points_np[..., 2].min(), points_np[..., 2].max())
        
        # Pre-calculate color limits for consistent coloring
        # Coloring by the X-coordinate (Dimension 0)
        color_data = points_np[..., 0] 
        c_lim = (color_data.min(), color_data.max())

        frames_for_gif = []

        for t in range(points_np.shape[0]):
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d') # This is the key line for 3D plotting
            
            frame_points = points_np[t] # Shape: (N, 3)

            # Create the 3D scatter plot
            sc = ax.scatter(
                frame_points[:, 0], # X axis
                frame_points[:, 1], # Y axis
                frame_points[:, 2], # Z axis
                c=frame_points[:, 0], # Color by X-axis value
                cmap='viridis',
                vmin=c_lim[0],
                vmax=c_lim[1]
            )
            
            # Set labels and title
            ax.set_xlabel('Dimension 0 (X)')
            ax.set_ylabel('Dimension 1 (Y)')
            ax.set_zlabel('Dimension 2 (Z)')
            ax.set_title(f'Frame {t} - 3D Anchor Trajectories')
            
            # Set consistent axis limits
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            
            # Add a color bar
            plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Dimension 0 Value')
            
            # Save the individual frame
            frame_path = os.path.join(traj_dir, f"frame_{t:04d}.png")
            plt.savefig(frame_path)
            plt.close(fig) # Close the figure to free up memory

            # Read the saved image to be added to the GIF
            frame_img = imageio.v2.imread(frame_path)
            frames_for_gif.append(frame_img)
        
        # Save the collected frames as a GIF
        gif_path = os.path.join(traj_dir, "trajectories_3d_video.gif")
        imageio.mimsave(gif_path, frames_for_gif, fps=8)
        dprint(f"Saved 3D trajectories GIF to {gif_path}")
            
    def _save_3d_trajectories_interactive(self, points_3d, save_dir):
        """Save 3D trajectories as an interactive HTML file for each frame using Plotly."""
        traj_dir = os.path.join(save_dir, "3d_trajectories_interactive")
        os.makedirs(traj_dir, exist_ok=True)
        dprint(f"Saving interactive 3D trajectories to {traj_dir}")

        points_np = points_3d.cpu().numpy() # Shape: (T, N, 3)

        # Pre-calculate axis limits across all frames for a stable view
        x_lim = (points_np[..., 0].min(), points_np[..., 0].max())
        y_lim = (points_np[..., 1].min(), points_np[..., 1].max())
        z_lim = (points_np[..., 2].min(), points_np[..., 2].max())
        
        # Pre-calculate color limits
        color_data = points_np[..., 0] # Using x-axis for color, like the original plot
        c_lim = (color_data.min(), color_data.max())

        for t in range(points_np.shape[0]):
            frame_points = points_np[t] # Shape: (N, 3)

            # Create an interactive 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=frame_points[:, 0],
                y=frame_points[:, 1],
                z=frame_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=frame_points[:, 0],  # Color by the x-coordinate
                    colorscale='Viridis',      # Use the same colormap
                    cmin=c_lim[0],             # Set consistent color range
                    cmax=c_lim[1],
                    colorbar=dict(
                        title="Dimension 0 (X-axis)"
                    ),
                    opacity=0.8
                )
            )])

            # Update layout for title and stable axis ranges
            fig.update_layout(
                title=f'Interactive 3D Trajectories - Frame {t}',
                scene=dict(
                    xaxis=dict(title='Dimension 0', range=x_lim),
                    yaxis=dict(title='Dimension 1', range=y_lim),
                    zaxis=dict(title='Dimension 2', range=z_lim),
                    aspectmode='data' # This makes the aspect ratio reflect the data scales
                ),
                margin=dict(r=20, b=10, l=10, t=40)
            )

            # Save to an HTML file
            html_path = os.path.join(traj_dir, f"interactive_frame_{t:04d}.html")
            fig.write_html(html_path)

        dprint(f"Finished saving {points_np.shape[0]} interactive HTML files.")

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
        if self.cfg.use_optimization:
            self.loss_history = {
                'l1': [],
                'lpips': [],
                'ssim': [],
                'total': []
            }
        
    def training_step(self, batch, batch_idx):
        self.actual_step += 1
        
        # ===================================================================
        # ===================== MAIN CONDITIONAL LOGIC ======================
        # ===================================================================
        
        if self.cfg.use_optimization:
            #---path 1: optimization training step ---
            opt = self.optimizers()
            opt.zero_grad()
            
            frame_idx = self.actual_step % self.target_video_frames.shape[0]
            time_t = float(frame_idx) / (self.target_video_frames.shape[0] - 1) if self.target_video_frames.shape[0] > 1 else 0.0
            from ..geometry.gaussian_4d_motion import Camera
            fovy = batch["fovy"][0] 
            fovx = batch["fovx"][0] if "fovx" in batch else fovy
            w2c, proj, cam_p = get_cam_info_gaussian(c2w=batch["c2w"][0], fovx=fovx, fovy=fovy, znear=batch.get("znear", 0.1), zfar=batch.get("zfar", 100.0))
            viewpoint_cam = Camera(
                FoVx = fovx,
                FoVy = fovy,
                image_width = batch["width"],
                image_height = batch["height"],
                world_view_transform = w2c,
                full_proj_transform = proj,
                camera_center = cam_p,
            )
            output = self.renderer(viewpoint_camera=viewpoint_cam, time=time_t, bg_color=self.background.env_color)
            rendered_image = output["render"].unsqueeze(0)
            target_image = self.target_video_frames[frame_idx].unsqueeze(0).to(rendered_image.device)
            
            loss_l1 = F.l1_loss(rendered_image, target_image)
            loss_lpips = self.lpips_loss(rendered_image.clamp(0, 1), target_image.clamp(0, 1)).mean()
            loss_ssim = 1.0 - ssim(rendered_image, target_image)
            
            total_loss = self.cfg.loss.lambda_l1 * loss_l1 + self.cfg.loss.lambda_lpips * loss_lpips + self.cfg.loss.lambda_ssim * loss_ssim
            
            self.loss_history['l1'].append(loss_l1.item())
            self.loss_history['lpips'].append(loss_lpips.item())
            self.loss_history['ssim'].append(loss_ssim.item())
            self.loss_history['total'].append(total_loss.item())
            
            self.log("loss/l1", loss_l1, prog_bar=True)
            self.log("loss/lpips", loss_lpips, prog_bar=True)
            self.log("loss/ssim", loss_ssim, prog_bar=True)
            self.log("loss/total", total_loss, prog_bar=True)
            
            if self.actual_step % 10 == 0 and self.actual_step > 0:
                import matplotlib.pyplot as plt
                
                steps = list(range(len(self.loss_history['l1'])))
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, self.loss_history['l1'], label=f"L1 (λ={self.cfg.loss.lambda_l1:.2f})")
                plt.plot(steps, self.loss_history['lpips'], label=f"LPIPS (λ={self.cfg.loss.lambda_lpips:.2f})")
                plt.plot(steps, self.loss_history['ssim'], label=f"SSIM (λ={self.cfg.loss.lambda_ssim:.2f})")
                plt.plot(steps, self.loss_history['total'], label="Total")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.title(f"Loss Curves up to Step {self.actual_step}")
                plt.legend()
                plt.grid(True)
                
                losses_dir = os.path.join(self.get_save_dir(), "losses")
                os.makedirs(losses_dir, exist_ok=True)
                plot_path = os.path.join(losses_dir, "loss_plot.png")
                plt.savefig(plot_path)
                plt.close()

            
            self.manual_backward(total_loss)
            opt.step()
            
            # After the optimizer updates the parameters, we must re-normalize the quaternions
            # to ensure they remain valid rotations.
            with torch.no_grad():
                self.geometry.rotation_delta.data = F.normalize(
                    self.geometry.rotation_delta.data, p=2, dim=-1
                )

            
            self.log("loss/l1", loss_l1, prog_bar=True)
            self.log("loss/lpips", loss_lpips, prog_bar=True)
            self.log("loss/ssim", loss_ssim, prog_bar=True)
            self.log("loss/total", total_loss, prog_bar=True)
            
            return {"loss": total_loss}
        
        else:
            # --- Path 2: Original Gaussians2Life Training Step ---
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
            # Extract num_frames to pass as positional argument and avoid conflict
            num_frames = batch["num_frames"]
            batch_kwargs = {k: v for k, v in batch.items() if k != "num_frames"}
            self.guidance.guidance_type = self.cfg.guidance_type
            with torch.no_grad():
                if hasattr(self.guidance, 'guidance_type') and self.guidance.guidance_type == 'ltx-video-guidance':
                    self.guidance(
                        guidance_inp, prompt_utils, num_frames, self.actual_step, self.trainer.max_steps,
                        zero_timestep=self.geometry.zero_timestep, rgb_as_latents=False, **batch_kwargs
                    )
                else:
                    self.guidance(
                        guidance_inp, prompt_utils, num_frames,
                        zero_timestep=self.geometry.zero_timestep, rgb_as_latents=False, **batch_kwargs
                    )
            dprint(f"1) Guidance time: {time.time() - start:.4f}")
            
            # Immediately clone the pristine, raw video from the guidance module
            # Check the format of diffusion_output and handle accordingly
            if hasattr(self.guidance, 'diffusion_output') and self.guidance.diffusion_output is not None:
                raw_video = self.guidance.diffusion_output.clone()
                dprint(f"Raw guidance video shape before processing: {raw_video.shape}")
                
                # LTX guidance outputs in [B, C, T, H, W] format
                # We need [B, T, C, H, W] for the saver
                if raw_video.dim() == 5 and raw_video.shape[1] == 3:  # Likely [B, C, T, H, W]
                    raw_guidance_video_for_saving = raw_video.permute(0, 2, 1, 3, 4)
                    dprint(f"Permuted from [B, C, T, H, W] to [B, T, C, H, W]: {raw_guidance_video_for_saving.shape}")
                else:
                    # Assume it's already in the correct format
                    raw_guidance_video_for_saving = raw_video
                    dprint(f"Using raw video as-is with shape: {raw_guidance_video_for_saving.shape}")
            else:
                dprint("WARNING: No diffusion_output found in guidance module")
                raw_guidance_video_for_saving = None

            # Save the raw video if debug artifacts are enabled
            if self.cfg.save_debug_artifacts and raw_guidance_video_for_saving is not None:
                save_dir = self._get_debug_save_dir(self.actual_step)
                self._save_video_as_frames_and_gif(raw_guidance_video_for_saving, save_dir, "raw_guidance_video")
                dprint(f"Saved CLEAN raw guidance video for step {self.actual_step}")
            
            # Save the raw video from DynamicRafter if available
            if hasattr(self.guidance, 'raw_video') and self.guidance.raw_video is not None:
                try:
                    save_dir = self._get_debug_save_dir(self.actual_step)
                    raw_video_dir = os.path.join(save_dir, "raw_video")
                    os.makedirs(raw_video_dir, exist_ok=True)
                    
                    # Save raw video as frames and GIF
                    raw_video = self.guidance.raw_video  # Should be in format (B, T, C, H, W) with values in [0, 1]
                    dprint(f"Raw video shape: {raw_video.shape}, dtype: {raw_video.dtype}, min: {raw_video.min().item():.3f}, max: {raw_video.max().item():.3f}")
                    frames_dir, gif_path = self._save_video_as_frames_and_gif(raw_video, raw_video_dir, "raw")
                    dprint(f"Saved raw video frames to {frames_dir} and GIF to {gif_path}")
                    
                    # Save v_s-1 (pre-warped video)
                    if hasattr(self.guidance, 'pre_warped_video_output'):       
                        # Shape is already (B, T, C, H, W)
                        pre_warped_for_saving = self.guidance.pre_warped_video_output.clone()
                        self._save_video_as_frames_and_gif(pre_warped_for_saving, save_dir, "v_s-1-pre_warp")
                        dprint(f"Saved PRE-WARPED video for step {self.actual_step}")
                    
                    # Save v'_s-1 (post-warped video)
                    if hasattr(self.guidance, 'post_warped_video_output'):
                        # Shape is already (B, T, C, H, W)
                        post_warped_for_saving = self.guidance.post_warped_video_output.clone()
                        self._save_video_as_frames_and_gif(post_warped_for_saving, save_dir, "v_s-1-post_warp")
                        dprint(f"Saved POST-WARPED video for step {self.actual_step}")
                            
                except Exception as e:
                    print(f"WARNING: Failed to save raw video. Error: {e} or")
                    print(f"WARNING: Failed to save pre-warped video. Error: {e} or")
                    print(f"WARNING: Failed to save post-warped video. Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save the generated guidance video for debugging
            if batch["view_index"] in self.guidance.output_stack:
                try:
                    # Get the video tensor (B, C, T, H, W) with values in [0, 1]
                    guidance_video_to_save = self.guidance.output_stack[batch["view_index"]]

                    # Format for saving: (T, H, W, C), uint8, CPU
                    formatted_video = (guidance_video_to_save[0].permute(1, 2, 3, 0) * 255).byte().cpu() # T H W C

                    # Define save subdirectory relative to the experiment's main save path
                    relative_save_subdir = os.path.join("debug", f"guidance_view_{batch['view_index']}")
                    # Get the full path where frames will be saved
                    full_save_dir = os.path.join(self.get_save_dir(), relative_save_subdir)
                    os.makedirs(full_save_dir, exist_ok=True) # Ensure directory exists

                    frame_filenames = [] # Keep track of saved frame filenames 

                    # Save individual frames first
                    for frame_idx, frame_tensor in enumerate(formatted_video):
                        frame_filename = f"frame_{frame_idx:04d}.png"
                        full_frame_path = os.path.join(full_save_dir, frame_filename)
                        # Use the system's save_image to handle path logic correctly (saves relative to get_save_dir())
                        self.save_image(
                        os.path.join(relative_save_subdir, frame_filename), # Path relative to main save dir
                        frame_tensor # Expects H W C
                        )
                        frame_filenames.append(full_frame_path) # Store the full path for imageio

                    # --- Manually compile the saved frames into a GIF ---
                    gif_filename = f"guidance_video_view_{batch['view_index']}_step{self.true_global_step}.gif"
                    full_gif_path = os.path.join(full_save_dir, gif_filename)

                    frames_for_gif = []
                    for fname in frame_filenames:
                        if os.path.exists(fname): # Check if frame was actually saved
                            frames_for_gif.append(imageio.v2.imread(fname))
                        else:
                            print(f"Warning: Frame not found for GIF compilation: {fname}")


                    if frames_for_gif: # Only proceed if we have frames
                        # Use imageio v2 API
                        imageio.v2.mimsave(full_gif_path, frames_for_gif, duration=(1000 / 8), loop=0) # fps=8 -> duration=125ms=1000/8
                        dprint(f"Saved guidance video frames to {full_save_dir} and compiled GIF to {full_gif_path}")
                    else:
                        print(f"Warning: No frames found to compile GIF for view {batch['view_index']}.")
                    # --- End of manual compilation ---

                except Exception as e:
                    print(f"WARNING: Failed to save guidance video/GIF for view {batch['view_index']}. Error: {e}")

            # Prepare guidance video for processing
            if hasattr(self.guidance, 'diffusion_output') and self.guidance.diffusion_output is not None:
                raw_video = self.guidance.diffusion_output
                dprint(f"Guidance video shape before processing: {raw_video.shape}")
                
                # LTX guidance outputs in [B, C, T, H, W] format
                # We need [B, T, C, H, W] for processing
                if raw_video.dim() == 5 and raw_video.shape[1] == 3:  # Likely [B, C, T, H, W]
                    guidance_video = raw_video.permute(0, 2, 1, 3, 4)
                    dprint(f"Permuted guidance video from [B, C, T, H, W] to [B, T, C, H, W]: {guidance_video.shape}")
                else:
                    # Assume it's already in the correct format
                    guidance_video = raw_video
                    dprint(f"Using guidance video as-is with shape: {guidance_video.shape}")
            else:
                dprint("ERROR: No diffusion_output found in guidance module")
                # Create a dummy tensor to avoid errors
                guidance_video = torch.zeros(1, self.cfg.geometry["num_frames"], 3, 
                                        out["comp_rgb"].shape[2], out["comp_rgb"].shape[3], 
                                        device=out["comp_rgb"].device)

            start = time.time()
            # find timestep which best aligns with no deformation timestep
            static_view = out["comp_rgb"][self.geometry.zero_timestep].permute(2, 0, 1)
            per_frame_diffs = torch.abs(guidance_video[0] - static_view).mean(dim=3).mean(dim=2).mean(dim=1)
            timestep = torch.argmin(per_frame_diffs).item()
            dprint(f"2) Find timestep time: {time.time() - start:.4f}, timestep: {timestep}")

            start = time.time()
            trajectories, depth_confidence, depth_maps, confidence_maps, pred_tracks, pred_visibility = self.flow_back_projection.back_project(
                guidance_video, out["rendered_depth"][self.geometry.zero_timestep, 0], timestep=timestep
            )
            dprint(f"3) Back projection time: {time.time() - start:.4f}")

            # Save debug artifacts if enabled
            if self.cfg.save_debug_artifacts:
                save_dir = self._get_debug_save_dir(self.actual_step)
                
                # Save static view
                static_view_path = os.path.join(save_dir, "static_view.png")
                save_image(static_view, static_view_path)
                
                # Save guidance video (this will now show the artifacts, which is good for debugging the tracker!)
                self._save_video_as_frames_and_gif(guidance_video, save_dir, "processed_guidance_video_with_artifacts")
                
                
                
                # Save depth maps and confidence maps
                self._save_depth_maps(depth_maps, confidence_maps, save_dir)
                
                # Save point tracks visualization
                self._save_point_tracks(guidance_video, pred_tracks, pred_visibility, save_dir)
                
                # Save 3D trajectories
                self._save_3d_trajectories(trajectories, save_dir)
                
                # Save 3D trajectories 
                self._save_3d_trajectories_interactive(trajectories, save_dir)

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
        # In optimization mode, we save the full rendered video during on_validation_epoch_end
        if self.cfg.use_optimization:
            return
        
        # original g2l val logic
        start = time.time()
        batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
        batch_video["frame_times"] = batch["frame_times_video"]
        out_video = self(batch_video, testing=True)
        self.save_video(out_video, f"{self.true_global_step}-eval")
        dprint(f"A) Evaluation time: {time.time() - start:.4f}")

    def on_validation_epoch_end(self):
        # This hook is used to render the full video in optimization mode.
        if self.cfg.use_optimization and self.trainer.global_step > 0:
            if self.true_global_step % 10 != 0:
                return
            threestudio.info(f"Rendering validation video at step {self.true_global_step}...")
            out_file = self.get_save_path(f"it{self.true_global_step}-val.mp4")
            writer = imageio.get_writer(out_file, fps=24)
            
            # Get the single fixed validation camera pose.
            batch = self.trainer.datamodule.val_dataloader().dataset[0]
            
            from ..geometry.gaussian_4d_motion import Camera
            fovy = batch["fovy"][0]; fovx = batch["fovx"][0] if "fovx" in batch else fovy
            w2c, proj, cam_p = get_cam_info_gaussian(c2w=batch["c2w"][0], fovx=fovx, fovy=fovy, znear=batch.get("znear", 0.1), zfar=batch.get("zfar", 100.0))
            viewpoint_cam = Camera(
                FoVx=fovx, FoVy=fovy, image_height=batch['height'], image_width=batch['width'],
                world_view_transform=w2c, full_proj_transform=proj, camera_center=cam_p
            )

            num_render_frames = self.target_video_frames.shape[0]
            for i in range(num_render_frames):
                time_t = float(i) / (num_render_frames - 1) if num_render_frames > 1 else 0.0
                with torch.no_grad():
                    out = self.renderer(viewpoint_camera=viewpoint_cam, time=time_t, bg_color=self.background.env_color)
                img = (out["render"].permute(1, 2, 0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                writer.append_data(img)
            writer.close()
            threestudio.info(f"Saved validation video to {out_file}")
        else:
            # original g2l val logic
            pass

    def test_step(self, batch, batch_idx):
        
        # For optimization mode, the validation logic is sufficient for testing.
        if self.cfg.use_optimization:
            self.on_validation_epoch_end()
            return
        
        #original g2l test logic:
        
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
