import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


########################################################################################################################
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


########################################################################################################################

# Adapated from https://github.com/JonathonLuiten/Dynamic3DGaussians/blob/7dbbd4dec404308524ff402756bdb8143a2589b0/train.py#L100C48-L100C63

def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def inv_rot(rot):
    # inverse rotation of normalized quaternion
    return rot * torch.tensor([1, -1, -1, -1], device=rot.device, dtype=rot.dtype)


def rigidity_loss(updated_pos, updated_rot, geometry):
    from ..geometry.utils import quat_mult, build_rotation

    num_frames = updated_pos.shape[0]
    prev_inv_rot = inv_rot(updated_rot[0])
    prev_offset = updated_pos[0][geometry.knn_indices] - updated_pos[0][:, None]
    loss_rigidity = 0.0
    for i in range(1, num_frames):
        curr_rot = updated_rot[i]
        rel_rot = quat_mult(curr_rot, prev_inv_rot)
        rot = build_rotation(rel_rot)
        curr_offset = updated_pos[i][geometry.knn_indices] - updated_pos[i][:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

        loss_rigidity += weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offset, geometry.knn_weights)

        prev_inv_rot = inv_rot(curr_rot)
        prev_offset = curr_offset

    loss_rigidity /= (num_frames - 1)
    return loss_rigidity


def rotation_similarity_loss(updated_rot, geometry):
    from ..geometry.utils import quat_mult

    num_frames = updated_rot.shape[0]
    prev_inv_rot = inv_rot(updated_rot[0])
    loss_rot_sim = 0.0
    for i in range(1, num_frames):
        curr_rot = updated_rot[i]
        rel_rot = quat_mult(curr_rot, prev_inv_rot)
        loss_rot_sim += weighted_l2_loss_v2(rel_rot[geometry.knn_indices], rel_rot[:, None], geometry.knn_weights)
        prev_inv_rot = inv_rot(curr_rot)

    loss_rot_sim /= (num_frames - 1)
    return loss_rot_sim


def longterm_isometry_loss(updated_pos, geometry):
    loss_longterm_iso = 0.0
    for i in range(1, len(updated_pos)):
        curr_offset = updated_pos[i][geometry.knn_indices] - updated_pos[i][:, None]
        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        loss_longterm_iso += weighted_l2_loss_v1(curr_offset_mag, torch.sqrt(geometry.knn_squared_dists),
                                                 geometry.knn_weights)

    loss_longterm_iso /= (len(updated_pos) - 1)
    return loss_longterm_iso


########################################################################################################################

# Inspired by SC-GS (https://yihua7.github.io/SC-GS-web/)
def arap_loss(updated_pos, geometry):
    neighbor_indices = geometry.knn_indices
    neighbor_weights = geometry.knn_weights
    num_frames = updated_pos.shape[0]
    import roma
    loss_arap = 0.0

    prev_offset = updated_pos[0][neighbor_indices] - updated_pos[0][:, None]
    for i in range(1, num_frames):
        curr_offset = updated_pos[i][neighbor_indices] - updated_pos[i][:, None]
        R_pred = roma.rigid_vectors_registration(prev_offset, curr_offset, neighbor_weights)
        loss_arap += weighted_l2_loss_v2(torch.matmul(prev_offset, torch.transpose(R_pred, -1, -2)), curr_offset,
                                         neighbor_weights)
        prev_offset = curr_offset

    loss_arap /= (num_frames - 1)
    return loss_arap


########################################################################################################################

# Inspired by Neural Parametric Gaussians
def simple_rigidity_loss(displacement, geometry):
    num_frames = displacement.shape[0]
    loss_rigidity = 0.0
    for i in range(num_frames):
        loss_rigidity += weighted_l2_loss_v2(displacement[i][geometry.knn_indices], displacement[i][:, None],
                                             geometry.knn_weights)
    loss_rigidity /= num_frames
    return loss_rigidity


########################################################################################################################

# Inspired by Align-Your-Gaussians
def jsd_loss(updated_pos):
    # compute means of the point clouds
    mean_pos = updated_pos.mean(dim=1)
    # compute diagonal covariance of the point clouds
    cov_pos = updated_pos.var(dim=1)

    # compute the JSD loss
    loss_jsd = torch.mean(torch.sum(
        -0.5 * torch.log(torch.tensor(2)) + 0.5 * torch.log(cov_pos[0] + cov_pos[1:]) - 0.25 * torch.log(
            cov_pos[0]) - 0.25 * torch.log(cov_pos[1:]) + 0.25 * (mean_pos[1:] - mean_pos[0]) ** 2 / (
            cov_pos[0] + cov_pos[1:]) + 0.25 * (mean_pos[0] - mean_pos[1:]) ** 2 / (cov_pos[0] + cov_pos[1:]),
        dim=1), dim=0)

    return loss_jsd


########################################################################################################################

# Inspired by GaussianFlow (https://arxiv.org/abs/2403.12365)
class FlowLoss:
    def __init__(self, device, model="small"):
        if model == "small":
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device)
        elif model == "large":
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.model = raft_large(weights=Raft_Large_Weights).to(device)
        else:
            raise ValueError("Invalid model name")
        self.model.eval()
        self.gt_flow = None
        self.gt_flow_library = {}

    def compute_gt_flow(self, video):
        video = video[0].permute(1, 0, 2, 3)
        # map [0, 1] to [-1, 1]
        video = 2 * video - 1
        # compute 2D flow
        gt_flow = self.model(video[:-1], video[1:])[-1]
        return gt_flow

    def compute(self, out, guidance_video, recompute_gt_flow=True, view_index=None):
        if view_index is not None:
            if view_index not in self.gt_flow_library:
                with torch.no_grad():
                    # compute 2D flow
                    gt_flow = self.compute_gt_flow(guidance_video)
                    self.gt_flow_library[view_index] = gt_flow
            self.gt_flow = self.gt_flow_library[view_index]
            gt_flow = self.gt_flow
        elif recompute_gt_flow or self.gt_flow is None:
            with torch.no_grad():
                # compute 2D flow
                gt_flow = self.compute_gt_flow(guidance_video)
                self.gt_flow = gt_flow
        else:
            gt_flow = self.gt_flow

        small_motion_mask = (torch.norm(gt_flow, p=2, dim=1) < 25)[:, None].expand(-1, 2, -1, -1)

        # compute gaussian flow
        gaussian_flow = []
        for i in range(len(out["comp_rgb"]) - 1):
            gaussian_flow.append(compute_gaussian_flow(out["2d_mean"][i], out["2d_cov_inv"][i], out["2d_mean"][i + 1],
                                                       out["2d_cov"][i + 1], out["indices"][i], out["x_mu"][i],
                                                       out["weights"][i]))
        gaussian_flow = torch.stack(gaussian_flow, dim=0)
        # compute loss
        loss = torch.norm(gt_flow - gaussian_flow, p=2, dim=1)[small_motion_mask[:, 0]].mean()
        # loss = l2_loss(gaussian_flow, gt_flow)
        return {"loss": loss, "gt_flow": gt_flow, "gaussian_flow": gaussian_flow,
                "small_motion_mask": small_motion_mask}


def compute_gaussian_flow(
    mean_2d_t,  # N x 2
    cov_2d_inv_t,  # N x 3
    mean_2d_t1,  # N x 2
    cov_2d_t1,  # N x 3
    indices,  # K x H x W
    x_mu,  # K x 2 x H x W
    weights  # K x H x W
):  # -> 2 x H x W

    # Detach tensors from first frame
    mean_2d_t = mean_2d_t.detach()
    cov_2d_inv_t = cov_2d_inv_t.detach()
    indices = indices.detach()
    x_mu = x_mu.detach()
    weights = weights.detach()

    # Resizing operations
    cov_2d_inv_t_x2 = cov_2d_inv_t[:, 0]
    cov_2d_inv_t_xy = cov_2d_inv_t[:, 1]
    cov_2d_inv_t_y2 = cov_2d_inv_t[:, 2]
    cov_2d_inv_t = torch.stack([cov_2d_inv_t_x2, cov_2d_inv_t_xy, cov_2d_inv_t_xy, cov_2d_inv_t_y2], dim=-1).view(-1, 2,
                                                                                                                  2)  # N x 2 x 2
    cov_2d_t1_x2 = cov_2d_t1[:, 0]
    cov_2d_t1_xy = cov_2d_t1[:, 1]
    cov_2d_t1_y2 = cov_2d_t1[:, 2]
    cov_2d_t1 = torch.stack([cov_2d_t1_x2, cov_2d_t1_xy, cov_2d_t1_xy, cov_2d_t1_y2], dim=-1).view(-1, 2,
                                                                                                   2)  # N x 2 x 2

    # Compute the flow
    return torch.sum(weights.unsqueeze(-1) * (
        (cov_2d_t1[indices] @ cov_2d_inv_t[indices] @ x_mu.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1) + (
        mean_2d_t1[indices] - mean_2d_t[indices] - x_mu.permute(0, 2, 3, 1))), dim=0).permute(2, 0, 1)
