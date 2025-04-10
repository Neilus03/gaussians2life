import torch

from scipy.special import erfinv
import numpy as np
import torch.nn.functional as F

def sample_points_dense_middle(n_points, start, end):
    # Generates sample points along a line with denser distribution in the middle using a Gaussian-like transformation.
    uniform_points = np.linspace(0, 1, n_points + 2)[1:-1]
    gaussian_like_points = np.sqrt(2) * erfinv(2 * uniform_points - 1)
    # gaussian_like_points = (np.abs(gaussian_like_points) ** 0.1) * gaussian_like_points
    gaussian_like_points = (gaussian_like_points - gaussian_like_points.min()) / (
        gaussian_like_points.max() - gaussian_like_points.min())
    dense_points = start + gaussian_like_points * (end - start)
    return dense_points


def sample_tracking_points(width, height, grid_size):
    """
    Sample tracking points on a grid that is denser in the center of the image.

    :param width: width of the image
    :param height: height of the image
    :param grid_size: number of points per axis
    :return: tensor of shape (N, 2) with N = grid_size^2
    """
    x = torch.tensor(sample_points_dense_middle(grid_size + 2, 0, width)[1:-1])
    y = torch.tensor(sample_points_dense_middle(grid_size + 2, 0, height)[1:-1])
    grid_x, grid_y = torch.meshgrid(x, y)
    grid = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
    # grid = torch.cat([torch.zeros((grid.size(0), 1)), grid], dim=1)[None]
    return grid


def interpolate_and_extrapolate(tensor, confidence, mask):
    """
    Interpolate missing values in tensor using linear interpolation. Extrapolate by repeating the first and last value.

    :param tensor: tensor of shape (T, N, C)
    :param confidence: tensor of shape (T, N) with confidence values for each value in tensor
    :param mask: tensor of shape (T, N) with 1 where values are valid and 0 where values are missing
    :return: tuple of two tensors of shape (T, N, C) with missing values interpolated / extrapolated
    """
    device = tensor.device
    tensor = tensor.cpu().numpy()  # Convert to numpy array for processing
    confidence = confidence.cpu().numpy()
    mask = mask.cpu().numpy()

    T, N, C = tensor.shape

    for n in range(N):
        for c in range(C):
            data = tensor[:, n, c]
            conf = confidence[:, n]
            mask_ = mask[:, n]

            if mask_.sum() == 0 or mask_.sum() == T:
                continue

            if mask_.sum() == 1:
                interpolated = np.full(T, data[mask_][0])
                tensor[:, n, c] = interpolated
                confidence[:, n] = conf[mask_].mean() / 6
                continue

            # Indices where values are not NaN
            indices = np.where(mask_)[0]
            values = data[mask_]

            # Interpolate missing values
            # interpolated = np.interp(np.arange(T), indices, values)
            from scipy.interpolate import CubicSpline

            def add_boundary_knots(spline):
                """
                Add knots infinitesimally to the left and right.

                Additional intervals are added to have zero 2nd and 3rd derivatives,
                and to maintain the first derivative from whatever boundary condition
                was selected. The spline is modified in place.
                """
                # determine the slope at the left edge
                leftx = spline.x[0]
                lefty = spline(leftx)
                leftslope = spline(leftx, nu=1)

                # add a new breakpoint just to the left and use the
                # known slope to construct the PPoly coefficients.
                leftxnext = np.nextafter(leftx, leftx - 1)
                leftynext = lefty + leftslope * (leftxnext - leftx)
                leftcoeffs = np.array([0, 0, leftslope, leftynext])
                spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

                # repeat with additional knots to the right
                rightx = spline.x[-1]
                righty = spline(rightx)
                rightslope = spline(rightx, nu=1)
                rightxnext = np.nextafter(rightx, rightx + 1)
                rightynext = righty + rightslope * (rightxnext - rightx)
                rightcoeffs = np.array([0, 0, rightslope, rightynext])
                spline.extend(rightcoeffs[..., None], np.r_[rightxnext])

            spline = CubicSpline(np.arange(T)[indices], values, bc_type="natural")
            add_boundary_knots(spline)
            interpolated = spline(np.arange(T))

            tensor[:, n, c] = interpolated
            confidence[~indices, n] = conf[indices].mean() / 4  # TODO: perhaps decrease confidence for these points

    return torch.from_numpy(tensor).to(device), torch.from_numpy(confidence).to(device)


class FlowBackProjection:

    def __init__(self, camera_intrinsics, device, **kwargs):
        self.device = device
        self.camera_intrinsics = camera_intrinsics.to(device)

        # Load depth model
        self.depth_model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vitl14",
                                          pretrained=True, trust_repo=True, force_reload=True).to(device)

        # Load tracking model
        self.point_tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        self.point_tracking_grid_size = kwargs.get("point_tracking_grid_size", 40)

        self.point_tracking_grid = None
        self.depth_threshold_factor = kwargs.get("depth_threshold_factor", 1.1)

    def get_depth(self, video):
        """
        Predict depth for video (per-frame metric depth prediction), processing frame-by-frame.

        :param video: tensor of shape (T, 3, H, W) with values in [0, 255]
        :return: tuple of tensors (pred_depth, pred_depth_confidence) with shapes (T, H, W) and (T, H, W)
        """
        num_frames, _, original_H, original_W = video.shape
        device = video.device
        all_pred_depth = []
        all_pred_confidence = []

        expected_H, expected_W = 364, 574 # Keep the resize target for now

        single_intrinsics = self.camera_intrinsics[None].to(device) # Shape [1, 3, 3]

        for t in range(num_frames):
            frame = video[t:t+1] # Get single frame, keep batch dim: [1, 3, H, W]

            # Resize input frame
            frame_resized = F.interpolate(frame, size=(expected_H, expected_W), mode='bilinear', align_corners=False)

            print(f"DEBUG: Inferring depth for frame {t}/{num_frames}, input shape: {frame_resized.shape}, intrinsics shape: {single_intrinsics.shape}")

            with torch.no_grad():
                # Pass single frame and single intrinsic matrix
                try:
                    # It expects batch size of 1 for intrinsics if input batch is 1
                    output = self.depth_model.infer(frame_resized, single_intrinsics)
                except Exception as e:
                    print(f"ERROR during UniDepth infer for frame {t}: {e}")
                    # Add dummy outputs to avoid breaking the loop, or re-raise
                    output = {"depth": torch.zeros(1, 1, expected_H, expected_W, device=device),
                            "confidence": torch.zeros(1, 1, expected_H, expected_W, device=device)}


            # Resize output depth and confidence back to the original video size
            pred_depth_t = F.interpolate(output["depth"], size=(original_H, original_W), mode='bilinear', align_corners=False)
            pred_confidence_t = F.interpolate(output["confidence"], size=(original_H, original_W), mode='bilinear', align_corners=False)

            all_pred_depth.append(pred_depth_t.squeeze(0)) # Remove batch dim
            all_pred_confidence.append(pred_confidence_t.squeeze(0)) # Remove batch dim

        # Stack results along the time dimension
        final_pred_depth = torch.stack(all_pred_depth, dim=0)       # Shape [T, 1, H, W] -> [T, H, W] after squeeze
        final_pred_confidence = torch.stack(all_pred_confidence, dim=0) # Shape [T, 1, H, W] -> [T, H, W] after squeeze

        return final_pred_depth.squeeze(1), final_pred_confidence.squeeze(1) # Ensure shape is [T, H, W]

    def get_point_tracks(self, video, timestep=0, **kwargs):
        """
        Predict point tracks for points in video.

        :param video: tensor of shape (B, T, 3, H, W) with values in [0, 255]
        :param kwargs: additional arguments for the point tracker
        :return: tuple of tensors (pred_tracks, pred_visibility) with shapes (B, T, N, 2) and (B, T, N)
        """
        if self.point_tracking_grid is None:
            self.point_tracking_grid = sample_tracking_points(video.size(-1), video.size(-2),
                                                              self.point_tracking_grid_size).to(self.device).float()
        pred_tracks = []
        pred_visibility = []
        with torch.no_grad():
            # for t in range(video.size(1)):
            for t in range(timestep, timestep + 1):
                point_tracking_grid = torch.cat(
                    [torch.ones((self.point_tracking_grid.size(0), 1), device=self.device) * t,
                     self.point_tracking_grid], dim=1)[None]
                pred_tracks_, pred_visibility_ = self.point_tracker(video, queries=point_tracking_grid,
                                                                    backward_tracking=True, **kwargs)
                pred_tracks.append(pred_tracks_)
                pred_visibility.append(pred_visibility_)

        pred_tracks = torch.cat(pred_tracks, dim=2)
        pred_visibility = torch.cat(pred_visibility, dim=2)
        return pred_tracks, pred_visibility  # B T N 2, B T N

    def back_project(self, video, init_depth, timestep=0, **kwargs):
        """
        Back-project motion to 3D. Use init_depth to adjust the scale of the depth.

        :param video: tensor of shape (1, T, 3, H, W) with values in [0, 1]
        :param init_depth: tensor of shape (H, W)
        :param kwargs: additional arguments for the back-projection
        :return: tensor of shape (T, N, 3) with N = point_tracking_grid_size^2; 3D motion of tracked points in camera space
        """

        # Transform video to [0, 255]
        video = (video * 255).round()
        H, W = video.size(-2), video.size(-1)

        # Compute depth per frame
        print("DEBUG: Video shape passed to get_depth:", video[0].shape) # Debugging line as when trying inference failed here.
        depth, confidence = self.get_depth(video[0])

        # Compute point tracks
        pred_tracks, pred_visibility = self.get_point_tracks(video, timestep=timestep, **kwargs)

        # clip tracks to image size
        pred_tracks[0, :, :, 0] = torch.clamp(pred_tracks[0, :, :, 0], 0, W - 1)
        pred_tracks[0, :, :, 1] = torch.clamp(pred_tracks[0, :, :, 1], 0, H - 1)

        gt_init_depths_per_point = init_depth[
            pred_tracks[0, timestep, :, 1].long(), pred_tracks[0, timestep, :, 0].long()]
        pred_depths_per_point = depth[0, pred_tracks[0, :, :, 1].long(), pred_tracks[0, :, :, 0].long()]
        depth_confidence_per_point = confidence[0, pred_tracks[0, :, :, 1].long(), pred_tracks[0, :, :, 0].long()]

        # find large jumps in depth and check whether depth at neighboring pixels is closer to previous depth
        for p in range(pred_depths_per_point.size(1)):
            depths = pred_depths_per_point[:, p][pred_visibility[0, :, p]]
            indices = torch.where(pred_visibility[0, :, p])[0]
            for t in range(1, len(depths)):
                if max(depths[t], depths[t - 1]) / min(depths[t], depths[t - 1]) > self.depth_threshold_factor:
                    # check neighboring pixels
                    window_size = 3
                    x = pred_tracks[0, indices[t], p, 0].long()
                    y = pred_tracks[0, indices[t], p, 1].long()
                    window = [max(y - window_size, 0),
                              min(y + window_size + 1, H),
                              max(x - window_size, 0),
                              min(x + window_size + 1, W)]

                    window_depth = depth[0, window[0]:window[1], window[2]:window[3]]

                    i = torch.argmin(torch.maximum(window_depth.flatten(), depths[t - 1]) /
                                     torch.minimum(window_depth.flatten(), depths[t - 1]))
                    d = window_depth.flatten()[i]
                    u = confidence[0, window[0]:window[1], window[2]:window[3]].flatten()[i]
                    if (max(d, depths[t - 1]) / min(d, depths[t - 1]) < max(depths[t], depths[t - 1]) /
                        min(depths[t], depths[t - 1])):
                        pred_depths_per_point[t, p] = d
                        # decrease confidence relative to distance (computation using i and manhattan distance)
                        depth_confidence_per_point[t, p] = u / (1 + abs(i % (2 * window_size + 1) - window_size) + abs(
                            i // (2 * window_size + 1) - window_size))
                        depths[t] = d

        # Interpolate depth values for where the points are not visible
        pred_depths_per_point, depth_confidence_per_point = interpolate_and_extrapolate(
            pred_depths_per_point.unsqueeze(-1),
            depth_confidence_per_point,
            pred_visibility[0, :, :])  # T N
        pred_depths_per_point = pred_depths_per_point.squeeze(-1)

        scale_factor = gt_init_depths_per_point / pred_depths_per_point[timestep]
        depths_per_point = pred_depths_per_point * scale_factor

        # Back-project points
        from .unidepth_utils import generate_rays, spherical_zbuffer_to_euclidean
        from einops import rearrange
        intrinsics = self.camera_intrinsics[None].repeat(video.size(1), 1, 1)
        angles = generate_rays(intrinsics, (H, W))[-1]
        angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
        angles_points = angles[0, :, pred_tracks[0, :, :, 1].long(), pred_tracks[0, :, :, 0].long()].permute(1, 2,
                                                                                                             0)  # T N 2
        depth_points = depths_per_point.unsqueeze(2)  # T N 1
        points_3d = torch.cat((angles_points, depth_points), dim=2)  # T N 3
        points_3d = spherical_zbuffer_to_euclidean(points_3d)  # T N 3

        return points_3d, depth_confidence_per_point
