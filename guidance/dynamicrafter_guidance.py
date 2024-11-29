import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *
from dataclasses import dataclass
import torch
import os
from collections import OrderedDict
import numpy as np
import random
from ..utils import dprint
import sys
import os.path as path


def add_path_to_dynamicrafter():
    HERE_PATH = path.normpath(path.dirname(__file__))
    DYNAMICRAFTER_PATH = path.normpath(path.join(HERE_PATH, './dynamicrafter'))
    # check the presence of models directory in repo to be sure its cloned
    if path.isdir(DYNAMICRAFTER_PATH):
        # workaround for sibling import
        sys.path.append(DYNAMICRAFTER_PATH)
    else:
        raise ImportError(f"DynamiCrafter is not initialized, could not find: {DYNAMICRAFTER_PATH}.\n "
                          "Did you forget to run 'git submodule update --init --recursive' ?")


def set_parameter_requires_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad

    for p in model.first_stage_model.parameters():
        p.requires_grad = requires_grad


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    dprint('>>> model checkpoint loaded.')
    return model


def make_ddim_timesteps(t, ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = max(int(num_ddpm_timesteps * t) // num_ddim_timesteps, 1)
        ddim_timesteps = np.asarray(list(range(0, max(int(num_ddpm_timesteps * t), num_ddim_timesteps), c)))
        steps_out = ddim_timesteps + 1
    elif ddim_discr_method == 'uniform_trailing':
        c = (num_ddpm_timesteps * t) / num_ddim_timesteps
        ddim_timesteps = np.flip(np.round(np.arange(num_ddpm_timesteps * t, 0, -c))).astype(np.int64)
        steps_out = ddim_timesteps - 1
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * t * .8), num_ddim_timesteps)) ** 2).astype(int)
        steps_out = ddim_timesteps + 1
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    if verbose:
        dprint(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_schedule(ddim_sampler, t, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
    ddim_sampler.ddim_timesteps = make_ddim_timesteps(t, ddim_discr_method=ddim_discretize,
                                                      num_ddim_timesteps=ddim_num_steps,
                                                      num_ddpm_timesteps=ddim_sampler.ddpm_num_timesteps,
                                                      verbose=verbose)
    alphas_cumprod = ddim_sampler.model.alphas_cumprod
    assert alphas_cumprod.shape[0] == ddim_sampler.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
    to_torch = lambda x: x.clone().detach().to(torch.float32).to(ddim_sampler.model.device)

    if ddim_sampler.model.use_dynamic_rescale:
        ddim_sampler.ddim_scale_arr = ddim_sampler.model.scale_arr[ddim_sampler.ddim_timesteps]
        ddim_sampler.ddim_scale_arr_prev = torch.cat(
            [ddim_sampler.ddim_scale_arr[0:1], ddim_sampler.ddim_scale_arr[:-1]])

    ddim_sampler.register_buffer('betas', to_torch(ddim_sampler.model.betas))
    ddim_sampler.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    ddim_sampler.register_buffer('alphas_cumprod_prev', to_torch(ddim_sampler.model.alphas_cumprod_prev))

    # calculations for diffusion q(x_t | x_{t-1}) and others
    ddim_sampler.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
    ddim_sampler.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
    ddim_sampler.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
    ddim_sampler.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
    ddim_sampler.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

    # ddim sampling parameters
    from .dynamicrafter.lvdm.models.utils_diffusion import make_ddim_sampling_parameters
    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                               ddim_timesteps=ddim_sampler.ddim_timesteps,
                                                                               eta=ddim_eta, verbose=verbose)
    ddim_sampler.register_buffer('ddim_sigmas', ddim_sigmas)
    ddim_sampler.register_buffer('ddim_alphas', ddim_alphas)
    ddim_sampler.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
    ddim_sampler.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
    sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
        (1 - ddim_sampler.alphas_cumprod_prev) / (1 - ddim_sampler.alphas_cumprod) * (
            1 - ddim_sampler.alphas_cumprod / ddim_sampler.alphas_cumprod_prev))
    ddim_sampler.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


@threestudio.register("dynamicrafter-guidance")
class DynamiCrafterGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_config: str = ""
        ckpt_path: str = ""
        anchor_viewpoint_sample_path: str = ""

        ddim_steps: int = 50
        ddim_eta: float = 1.0
        bs: int = 1
        height: int = 320
        width: int = 512
        num_frames: int = 16
        multiple_cond_cfg: bool = False
        cfg_img: float = 1.0
        timestep_spacing: str = "uniform_trailing"
        perframe_ae: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_scheduling: str = "linear"
        guidance_scale: float = 7.5

        frame_stride: int = 15
        guidance_rescale: float = 0.0

        lambda_latent_interpolation: Any = 0.0
        latent_interpolation_type: str = "previous"
        lambda_latent_accumulation: float = 0.7

        warping_mode: str = "flow"

    cfg: Config

    def configure(self) -> None:

        add_path_to_dynamicrafter()

        from omegaconf import OmegaConf
        ## model config
        config = OmegaConf.load(self.cfg.model_config)
        model_config = config.pop("model", OmegaConf.create())

        from .dynamicrafter.utils.utils import instantiate_from_config
        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.perframe_ae = self.cfg.perframe_ae
        print("current path", os.getcwd())
        assert os.path.exists(self.cfg.ckpt_path), "Error: Checkpoint Not Found!"
        self.model = load_model_checkpoint(model, self.cfg.ckpt_path)
        self.model.eval()

        ## run over data
        assert (self.cfg.height % 16 == 0) and (
            self.cfg.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        assert self.cfg.bs == 1, "Current implementation only support [batch size = 1]!"

        set_parameter_requires_grad(self.model, False)

        from .dynamicrafter.lvdm.models.samplers.ddim import DDIMSampler
        from .dynamicrafter.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond

        self.ddim_sampler = DDIMSampler(self.model) if not self.cfg.multiple_cond_cfg else DDIMSampler_multicond(model)

        self.num_train_timesteps = self.cfg.ddim_steps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.ddim_sampler.model.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info("Loaded Dynamicrafter Model!")

        if isinstance(self.cfg.lambda_latent_interpolation, float):
            self.lambda_latent_interpolation = self.cfg.lambda_latent_interpolation

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 512"]
    ) -> Float[Tensor, "B 4 40 64"]:

        from einops import rearrange
        def get_latent_z(model, videos):
            b, c, t, h, w = videos.shape
            x = rearrange(videos, 'b c t h w -> (b t) c h w')
            z = model.encode_first_stage(x)
            z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
            return z

        return get_latent_z(self.model, imgs)

    def compute_denoised(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        cond,
        u_cond,
        latents_prev=None,
        **kwargs
    ):
        if "index" in kwargs:
            kwargs.pop("index")
        with torch.no_grad():
            size = latents.shape
            samples, intermediates = self.ddim_sampler.ddim_sampling(cond, size,
                                                                     callback=None,
                                                                     img_callback=None,
                                                                     quantize_denoised=False,
                                                                     mask=None, x0=None,
                                                                     ddim_use_original_steps=False,
                                                                     noise_dropout=0.,
                                                                     temperature=1.,
                                                                     score_corrector=None,
                                                                     corrector_kwargs=None,
                                                                     x_T=latents,
                                                                     log_every_t=100,
                                                                     unconditional_guidance_scale=self.cfg.guidance_scale,
                                                                     unconditional_conditioning=u_cond,
                                                                     verbose=False,
                                                                     precision=None,
                                                                     fs=torch.tensor(
                                                                         [self.cfg.frame_stride] * self.cfg.bs,
                                                                         dtype=torch.long, device=self.device),
                                                                     guidance_rescale=self.cfg.guidance_rescale,
                                                                     **kwargs)
            diffusion_output = self.model.decode_first_stage(samples)  # B C T H W
        return diffusion_output

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompts: List[str],
        num_frames: int = 16,
        rgb_prev: Optional[Float[Tensor, "B H W C"]] = None,
        zero_timestep: int = 0,
        **kwargs,
    ):
        if "clear_previous" in kwargs and kwargs["clear_previous"]:
            self.output_stack = {}
            self.first_frames = {}
            self.first_frame_indices = {}
            self.current_guidance_view = None

        rgb = rgb.permute(0, 3, 1, 2)
        batch_size = rgb.shape[0] // num_frames
        import torch.nn.functional as F
        rgb = F.interpolate(rgb, (self.cfg.height, self.cfg.width), mode="bilinear", align_corners=False)[None].permute(
            0, 2, 1, 3, 4)  # B(1) C T H W

        img = rgb[:, :, zero_timestep]

        if hasattr(self, "load_anchor_viewpoint") and self.load_anchor_viewpoint:
            self.load_anchor_viewpoint = False

            import torchvision.io.video as video

            # Read the video frames
            video_tensor, _, _ = video.read_video(self.cfg.anchor_viewpoint_sample_path, pts_unit='sec')

            # if necessary, scale video
            if video_tensor.shape[0] > num_frames:
                video_tensor = video_tensor[:, :num_frames]
            if video_tensor.shape[1] != self.cfg.height or video_tensor.shape[2] != self.cfg.width:
                dprint(f">>> Resizing video from anchor viewpoint. (from shape {video_tensor.shape}).")
                from torchvision.transforms import Resize
                video_tensor = Resize((self.cfg.height, self.cfg.width))(video_tensor.permute(0, 3, 1, 2))[None]
                dprint(f">>> Resized to shape {video_tensor.shape}.")
            else:
                video_tensor = video_tensor.permute(0, 3, 1, 2)[None]
            self.diffusion_output = video_tensor.permute(0, 2, 1, 3, 4).to(self.device) / 255.0
            self.anchor_guidance_output = self.diffusion_output
            dprint(">>> Video from anchor viewpoint loaded.")

            if "view_index" in kwargs:
                self.output_stack[kwargs["view_index"]] = self.diffusion_output
                self.first_frames[kwargs["view_index"]] = rgb[0, :, 0].permute(1, 2, 0).detach().cpu().numpy()
                per_frame_diffs = torch.abs(
                    self.output_stack[kwargs["view_index"]][0].permute(1, 0, 2, 3).cpu() - torch.tensor(self.first_frames[
                        kwargs["view_index"]]).permute(2, 0, 1)).mean(dim=3).mean(dim=2).mean(dim=1)
                timestep = torch.argmin(per_frame_diffs).item()
                self.first_frame_indices[kwargs["view_index"]] = timestep

        elif "view_index" in kwargs and kwargs["view_index"] not in self.output_stack:

            with torch.no_grad():
                img_emb = self.model.embedder(img)  # B L C
                img_emb = self.model.image_proj_model(img_emb)
                cond_emb = self.model.get_learned_conditioning(prompts)

            cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

            latents = self.encode_images(rgb)

            img_cat_cond = latents[:, :, zero_timestep:zero_timestep + 1, :, :].repeat(1, 1, num_frames, 1, 1).detach()
            img_cat_cond.requires_grad = False
            cond["c_concat"] = [img_cat_cond]

            if self.cfg.guidance_scale != 1.0:
                if self.model.uncond_type == "empty_seq":
                    prompts = batch_size * [""]
                    uc_emb = self.model.get_learned_conditioning(prompts)  # TODO: Implement prompt preprocessing
                elif self.model.uncond_type == "zero_embed":
                    uc_emb = torch.zeros_like(cond_emb)
                uc_img_emb = self.model.embedder(torch.zeros_like(img))  ## b l c
                uc_img_emb = self.model.image_proj_model(uc_img_emb)
                uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
                if self.model.model.conditioning_key == 'hybrid':
                    uc["c_concat"] = [img_cat_cond]
            else:
                uc = None

            if self.cfg.multiple_cond_cfg and self.cfg.cfg_img != 1.0:
                uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
                if self.model.model.conditioning_key == 'hybrid':
                    uc_2["c_concat"] = [img_cat_cond]
                kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
            else:
                kwargs.update({"unconditional_conditioning_img_nonetext": None})

            from .dynamicrafter.lvdm.common import noise_like

            if rgb_prev is not None:
                rgb_prev = rgb_prev.permute(0, 3, 1, 2)
            elif "reference_view" in kwargs and kwargs["reference_view"] is not None:
                if "reference_view" == 0 and hasattr(self, "running_mean"):
                    self.running_mean = None
                rgb_prev = self.output_stack[kwargs["reference_view"]][0].permute(1, 0, 2, 3)
                dprint(f"Use reference view {kwargs['reference_view']} for guidance for {kwargs['view_index']}")

                if self.cfg.warping_mode == "flow":
                    from ..utils import warp_video_with_flow
                    rgb_prev = warp_video_with_flow(self.first_frames[kwargs["reference_view"]],
                                                    rgb[0, :, zero_timestep].permute(1, 2, 0).detach().cpu().numpy(),
                                                    rgb_prev, self.first_frame_indices[kwargs["reference_view"]])
                elif self.cfg.warping_mode == "homography":
                    from ..utils import transform_video_with_homography, find_homography
                    H = find_homography(self.first_frames[kwargs["reference_view"]],
                                        rgb[0, :, zero_timestep].permute(1, 2, 0).detach().cpu().numpy())
                    rgb_prev = transform_video_with_homography(rgb_prev, H)
                else:
                    raise ValueError(f"Unknown warping mode: {self.cfg.warping_mode}")

            from ..utils import DEBUG
            if DEBUG:
                video = rgb_prev.permute(0, 2, 3, 1).cpu().numpy()
                video = (video * 255).astype(np.uint8)
                import imageio
                imageio.mimsave(f"outputs/it{kwargs['view_index']}-cond.gif", video, fps=8)


            elif (hasattr(self, "anchor_guidance_output") and self.anchor_guidance_output is not None
                  and self.cfg.latent_interpolation_type == "anchor"):
                rgb_prev = self.anchor_guidance_output[0].permute(1, 0, 2, 3)
            elif (hasattr(self, "diffusion_output") and self.diffusion_output is not None
                  and self.cfg.latent_interpolation_type in ["previous", "running-mean"]):
                rgb_prev = self.diffusion_output[0].permute(1, 0, 2, 3)

            if self.lambda_latent_interpolation > 0 and rgb_prev is not None:
                rgb_prev = F.interpolate(rgb_prev, (self.cfg.height, self.cfg.width), mode="bilinear",
                                         align_corners=False)[None].permute(0, 2, 1, 3, 4)
                latents_prev = self.encode_images(rgb_prev)

                latents = ((1 - self.lambda_latent_interpolation) * latents +
                           self.lambda_latent_interpolation * latents_prev)

            noise = noise_like(latents.shape, device=self.device)

            make_schedule(self.ddim_sampler, t=self.curr_t_float,
                          # ddim_num_steps=int(self.cfg.ddim_steps * (1 - self.curr_t_float)),
                          ddim_num_steps=self.cfg.ddim_steps,
                          ddim_discretize=self.cfg.timestep_spacing,
                          ddim_eta=self.cfg.ddim_eta, verbose=False)

            latents = self.model.q_sample(latents,
                                          torch.tensor(self.ddim_sampler.ddim_timesteps[-1], device=latents.device)[
                                              None], noise=noise)

            latents = latents.detach()

            if self.lambda_latent_interpolation > 0 and rgb_prev is not None:
                self.diffusion_output = self.compute_denoised(latents, cond, uc, latents_prev=latents_prev,
                                                              **kwargs).detach()
            else:
                self.diffusion_output = self.compute_denoised(latents, cond, uc, **kwargs).detach()
            if "view_index" in kwargs:
                self.output_stack[kwargs["view_index"]] = self.diffusion_output
                self.first_frames[kwargs["view_index"]] = rgb[0, :, zero_timestep].permute(1, 2,
                                                                                           0).detach().cpu().numpy()
                self.first_frame_indices[kwargs["view_index"]] = zero_timestep
                dprint(">>> Multistep Update")

        if "view_index" in kwargs:
            if self.current_guidance_view != kwargs["view_index"]:
                self.diffusion_output = self.output_stack[kwargs["view_index"]]
                self.current_guidance_view = kwargs["view_index"]

        assert self.diffusion_output is not None, "Error: Diffusion Output is None!"
        ret = {
            "pseudo_loss": 0.0
        }
        return ret

    def update_t_w(self, step, max_steps):
        if self.cfg.diffusion_scheduling == "uniform":
            curr_t_float = random.uniform(self.cfg.min_step_percent, self.cfg.max_step_percent)
            curr_t = round(curr_t_float * self.num_train_timesteps)
        elif self.cfg.diffusion_scheduling == "linear":
            curr_t_float = self.cfg.max_step_percent - (step / max_steps) * (
                self.cfg.max_step_percent - self.cfg.min_step_percent)
            curr_t = round(curr_t_float * self.num_train_timesteps)
        else:
            raise ValueError(f"Unknown diffusion scheduling: {self.cfg.diffusion_scheduling}")

        self.curr_t_float = curr_t_float
        self.curr_t = curr_t

        if step == 0 and self.cfg.anchor_viewpoint_sample_path != "":
            self.load_anchor_viewpoint = True

        if hasattr(self.cfg.lambda_latent_interpolation, "__getitem__"):
            self.lambda_latent_interpolation = self.cfg.lambda_latent_interpolation[0] + (
                self.cfg.lambda_latent_interpolation[1] - self.cfg.lambda_latent_interpolation[
                0]) * step / max_steps
