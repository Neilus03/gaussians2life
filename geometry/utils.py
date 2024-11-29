import sys
from datetime import datetime
import random
import open3d as o3d
import numpy as np
import roma
import torch

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


########################################################################################################################

def o3d_knn(pts: torch.Tensor,
            num_knn: int = 10,
            src_pts: torch.Tensor = None):
    """
    Find the nearest neighbors of each point in a point cloud using Open3D.
    [Adopted from https://github.com/JonathonLuiten/Dynamic3DGaussians]

    :param pts: (m x 3) Point cloud with Gaussian centers
    :param num_knn: Number of nearest neighbors to find for each point
    :return: np.array (m x num_knn) of indices of nearest neighbors, np.array (m x num_knn x 3) of differences between
        each point and its neighbors, np.array (m x num_knn) of squared distances between each point and its neighbors
    """
    device = pts.device
    pts = pts.cpu().detach().numpy()
    indices = []
    diffs = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))

    if src_pts is not None:
        src_pts = src_pts.cpu().detach().numpy()
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(np.ascontiguousarray(src_pts, np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_src)
    else:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for n, p in enumerate(pcd.points):
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        if src_pts is not None:
            indices.append(i[:-1])
            diffs.append(src_pts[i[:-1]] - pts[n])
            sq_dists.append(d[:-1])
        else:
            indices.append(i[1:])
            diffs.append(pts[i[1:]] - pts[i[0]])
            sq_dists.append(d[1:])
    return (torch.tensor(indices, device=device).int(),
            torch.tensor(diffs, device=device).float(),
            torch.tensor(sq_dists, device=device).float())


def get_points_inside_bbox(points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    bbox = bbox.cpu().numpy()
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox))
    indices = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points.cpu().numpy()))
    return indices


def o3d_get_scene_transform_from_bbox(bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # compute the rotation and translation necessary to align the scene with the bbox (centered at origin)
    bbox = bbox.cpu().numpy()
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox))
    bbox_center = torch.tensor(bbox.center)
    bbox_rot = bbox.R.T
    bbox_rot = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(torch.tensor(bbox_rot)))
    bbox_extent = bbox.extent.max()
    return bbox_center, bbox_rot, bbox_extent


def transform_scene_from_bbox(_xyz: torch.Tensor, _rotation: torch.Tensor, _scaling: torch.Tensor, bbox: torch.Tensor,
                              rot_x=0, rot_y=0, rot_z=0):
    # TODO: Add support for SH coefficients
    bbox_center, bbox_rot, bbox_extent = o3d_get_scene_transform_from_bbox(bbox)
    bbox_center = bbox_center.to(_xyz.device).type(_xyz.dtype)
    bbox_rot = roma.quat_wxyz_to_xyzw(bbox_rot).to(_xyz.device).type(_xyz.dtype)
    if rot_x > 0 or rot_y > 0 or rot_z > 0:
        R = roma.euler_to_unitquat("XYZ", [rot_x, rot_y, rot_z], degrees=True, dtype=_xyz.dtype, device=_xyz.device)
        bbox_rot = roma.quat_product(R, bbox_rot)
    bbox_rot = bbox_rot[None, :]

    _xyz = _xyz - bbox_center
    _xyz = roma.quat_action(bbox_rot, _xyz, is_normalized=True)
    _xyz = _xyz / bbox_extent

    _rotation = roma.quat_xyzw_to_wxyz(roma.quat_product(bbox_rot, roma.quat_wxyz_to_xyzw(_rotation)))

    import math
    _scaling = _scaling - math.log(bbox_extent)

    return _xyz, _rotation, _scaling


def transform_scene_with_vector(_xyz: torch.Tensor, _rotation: torch.Tensor, _scaling: torch.Tensor,
                                vector: torch.Tensor):
    import math
    # Normalize the viewpoint to get the direction and distance
    viewpoint_norm = torch.norm(vector)
    direction = vector / viewpoint_norm

    # Compute elevation angle in degrees
    elevation = torch.asin(direction[2]).item() * 180.0 / math.pi

    # Compute azimuth angle
    azimuth = torch.atan2(direction[1], direction[0]).item() * 180.0 / math.pi + 90.0

    print("Azimuth: ", azimuth, "Elevation: ", elevation)

    # Construct the rotation matrix to align viewpoint with +X axis (azimuth=0)
    R_azimuth = roma.euler_to_unitquat("Z", [-azimuth], degrees=True, dtype=_xyz.dtype, device=_xyz.device)

    # Apply rotation to _xyz
    _xyz = roma.quat_action(R_azimuth[None, :], _xyz, is_normalized=True)
    _xyz = _xyz / viewpoint_norm

    # Apply rotation to _rotation
    _rotation = roma.quat_xyzw_to_wxyz(roma.quat_product(R_azimuth[None, :], roma.quat_wxyz_to_xyzw(_rotation)))

    # Scale the scene to make the distance to the viewpoint equal to 1
    _scaling = _scaling - math.log(viewpoint_norm)

    return _xyz, _rotation, _scaling, elevation


def quat_mult(q1, q2):
    """
    Multiply two tensors of quaternions. [Adopted from https://github.com/JonathonLuiten/Dynamic3DGaussians]

    :param q1: (n x 4) Tensor of quaternions
    :param q2: (n x 4) Tensor of quaternions
    :return: (n x 4) Tensor of quaternions
    """
    # w1, x1, y1, z1 = q1.T
    # w2, x2, y2, z2 = q2.T
    # w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    # x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    # y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    # z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    # return torch.stack([w, x, y, z]).T

    q1 = roma.quat_wxyz_to_xyzw(q1)
    q2 = roma.quat_wxyz_to_xyzw(q2)
    q_n = roma.quat_product(q1, q2)
    q_n = roma.quat_xyzw_to_wxyz(q_n)
    return q_n


def gaussian_kernel(x: torch.Tensor, sigma: float = 1.0):
    """
    Compute the Gaussian kernel for each element in the input tensor.

    :param x: Tensor of data points
    :param sigma: Sigma parameter for Gaussian kernel
    :return: x with Gaussian kernel applied
    """
    return torch.exp(-torch.sum(x ** 2, dim=1) / (2 * sigma ** 2))


def estimate_R_s_from_neighborhood_displacement(point_displacement: torch.Tensor,
                                                knn_relative_positions: torch.Tensor,
                                                knn_displacements: torch.Tensor):
    """
    For each point, infer the rotation and scaling from the displacement of its initial neighborhood.

    :param point_displacement: (m x 3) Displacement of each point (output of neural field)
    :param knn_relative_positions: (m x n x 3) Relative positions of each neighbor to the point (m points, n neighbors)
    :param knn_displacements: (m x n x 3) Displacements of each neighbor (output of neural field)
    :return: Rotation matrices for each Gaussian (m x 3 x 3) and scaling factors (m x 3)
    """
    displaced_positions = knn_relative_positions + knn_displacements - point_displacement[:, None, :]  # (m x n x 3)
    r, s = analytical_registration_batch(knn_relative_positions, displaced_positions)
    r = rot_to_quat_batched(r)
    return r, s


def analytical_registration_batch(points_in, points_out):
    """
    Batched version to analytically compute the rotation and scaling to register points_in to points_out.
    Implementation of the paper "A generalized closed-form solution for 3D registration of two-point sets under
    isotropic and anisotropic scaling" (https://doi.org/10.1016/j.rinp.2023.106746)

    :param points_out: (batch_dimension(s) x m x 3) Points to register
    :param points_in: (batch_dimension(s) x m x 3) Target points
    :return: Rotation matrices (batch_dimension(s) x 3 x 3) and scaling factors (batch_dimension(s) x 3)
    """
    points_out = points_out.transpose(-1, -2)
    points_in = points_in.transpose(-1, -2)

    points_out_points_out_T_inv = torch.linalg.inv(torch.matmul(points_in, points_in.transpose(-1, -2)))

    u, _, vh = torch.linalg.svd(torch.matmul(torch.matmul(points_out, points_in.transpose(-1, -2)),
                                             points_out_points_out_T_inv))
    det = torch.det(torch.matmul(u, vh).transpose(-1, -2))
    d = torch.eye(3, device=points_out.device).repeat(points_out.shape[:-2] + (1, 1))
    d[..., -1, -1] = det
    r = torch.matmul(torch.matmul(u, d), vh)

    s = torch.zeros(points_out.shape[:-2] + (3,), device=points_out.device)
    for i in range(3):
        i_ = torch.zeros((3, 3), device=points_out.device)
        i_[i, i] = 1
        numerator = torch.einsum('bji,bjk,kl,bli->b', points_out, r, i_, points_in)
        # numerator = torch.vmap(torch.trace)(torch.matmul(torch.matmul(points_out, r), i_[None]).matmul(points_in))
        denominator = torch.einsum('bji,jk,bki->b', points_in, i_, points_in)
        # denominator = torch.vmap(torch.trace)(torch.matmul(torch.matmul(points_in, i_[None]), points_in))
        s[..., i] = numerator / denominator

    return r, s


########################################################################################################################

# import unittest
# import torch
#
#
# def random_rotation_matrix():
#     random_matrix = torch.randn(3, 3)
#     q, r = torch.linalg.qr(random_matrix)
#
#     # Ensure the determinant is 1 (for a proper rotation matrix)
#     if torch.det(q) < 0:
#         q[:, -1] = -q[:, -1]  # Flip last column if determinant is -1
#
#     return q
#
#
# class TestNeighborhoodDisplacement(unittest.TestCase):
#     def setUp(self):
#         torch.manual_seed(0)  # For reproducible results
#
#     def test_analytical(self):
#         points = torch.randn(2000, 10, 3)
#         gt_scaling = torch.diag(torch.abs(torch.rand(3)))  # Random scaling
#         points_scaled = torch.einsum('ij, bmj -> bmi', gt_scaling, points)
#         gt_rotation = random_rotation_matrix()
#         points_scaled_rot = torch.einsum('ij, bmj -> bmi', gt_rotation, points_scaled)
#         # points_scaled_rot = torch.randn(2000,10, 3)
#
#         # rotation_matrix, scaling_factors = analytical_registration_batch(points.repeat(20,1,1), points_scaled_rot.repeat(20,1,1))
#         rotation_matrix, scaling_factors = analytical_registration_batch(points, points_scaled_rot)
#
#         self.assertTrue(torch.allclose(gt_scaling.diagonal(), scaling_factors, atol=1e-3))
#         self.assertTrue(torch.allclose(gt_rotation, rotation_matrix, atol=1e-3))
#
#     def test_backward(self):
#         points = torch.randn(2000, 10, 3).requires_grad_(True)
#         gt_scaling = torch.diag(torch.abs(torch.rand(3)))
#         points_scaled = torch.einsum('ij, bmj -> bmi', gt_scaling, points)
#         gt_rotation = random_rotation_matrix()
#         points_scaled_rot = torch.einsum('ij, bmj -> bmi', gt_rotation, points_scaled)
#
#         rotation_matrix, scaling_factors = analytical_registration_batch(points, points_scaled_rot)
#         loss = torch.sum(scaling_factors) + torch.sum(rotation_matrix)
#         try:
#             loss.backward()
#             self.assertTrue(points.grad is not None)
#             print(points.grad)
#             opt = torch.optim.SGD([points], lr=0.1)
#             opt.step()
#         except Exception as e:
#             self.fail(f"Backward pass failed with exception: {e}")
#
#
# if __name__ == '__main__':
#     unittest.main(argv=[''], exit=False)


########################################################################################################################

def interpolate_updates(anchor_updates: dict[str, torch.Tensor],
                        anchor_positions: torch.Tensor,
                        query_positions: torch.Tensor,
                        bandwidth: float = 1.0) -> dict[str, torch.Tensor]:
    """
    Takes a list of queried updates and interpolates them for other positions in the field while preserving the
    differentiability using Gaussian kernels.

    :param anchor_updates: A dictionary of updates for each sampled anchor position in the field.
    :param anchor_positions: The positions of the anchor points.
    :param query_positions: The positions for which to interpolate the updates.
    :param bandwidth: The bandwidth for the Gaussian kernel.
    :return: Interpolated updates for all positions in the field.
    """

    # Calculate distances between every query position and update position
    distances = torch.cdist(anchor_positions, query_positions)

    # Apply the Gaussian kernel to these distances
    weights = torch.exp_(-0.5 * (distances / bandwidth) ** 2)

    # Normalize the weights along the update dimension
    weights /= weights.sum(dim=1, keepdim=True)

    interpolated_updates = {key: torch.zeros(query_positions.shape[0], *value.shape[1:], device=value.device)
                            for key, value in anchor_updates.items()}

    for key in anchor_updates.keys():
        # Extract the update tensor for the current key from each update dictionary
        updates_tensor = anchor_updates[key]

        # Compute weighted updates in an optimized manner for the current tensor
        interpolated_update = torch.einsum('ij, ik -> jk', weights, updates_tensor)

        # Store the interpolated update in the dictionary
        interpolated_updates[key] = interpolated_update

    return interpolated_updates


def interpolate_changes_from_anchors(changes: dict[str, torch.Tensor],
                                     anchor_indices: torch.Tensor,  # (n x k)
                                     anchor_weights: torch.Tensor,  # (n x k)
                                     ):  # -> dict[str, torch.Tensor]:
    """
    Interpolates changes from anchor points to all points in the scene.

    :param changes: Changes at anchor points
    :param anchor_indices: Indices of closest k anchor points for each point in the scene
    :param anchor_weights: Weights of closest k anchor points for each point in the scene
    :return: Interpolated changes for all points in the scene
    """

    interpolated_changes = {}
    for key, value in changes.items():  # (m x d)
        # Extract the update tensor for the current key from each update dictionary
        updates_tensor = value[anchor_indices]  # (n x k x d)

        # Compute weighted updates in an optimized manner for the current tensor
        interpolated_update = torch.einsum('ijk, ij -> ik', updates_tensor, anchor_weights)

        # Store the interpolated update in the dictionary
        interpolated_changes[key] = interpolated_update

    return interpolated_changes


def estimate_rigid_transform(anchor_deformations, anchor_indices, anchor_weights, anchor_offsets, point_offsets=None):
    init_pcd = anchor_offsets
    step_pcd = anchor_offsets + anchor_deformations[anchor_indices]  # (n x k x 3)
    if point_offsets is not None:
        step_pcd = step_pcd - point_offsets[:, None, :]
    from roma import rigid_points_registration, rotmat_to_unitquat, quat_xyzw_to_wxyz
    R, T, s = rigid_points_registration(init_pcd, step_pcd, anchor_weights, compute_scaling=True)
    R = quat_xyzw_to_wxyz(rotmat_to_unitquat(R))
    # repeat s three times for 3 dimensions
    s = s[:, None].repeat(1, 3)
    return {
        "displacement": T,
        "rotation": R,
        "scale": s,
    }


def only_estimate_displacement(anchor_deformations, anchor_indices, anchor_weights, anchor_offsets, point_offsets=None):
    displacement = anchor_deformations[anchor_indices] * anchor_weights[..., None]
    displacement = displacement.sum(dim=1)

    rotation = torch.zeros((displacement.shape[0], 4), device=displacement.device)
    rotation[:, 0] = 1.0

    scale = torch.ones((displacement.shape[0], 3), device=displacement.device)

    return {
        "displacement": displacement,
        "rotation": rotation,
        "scale": scale
    }


########################################################################################################################

def rot_to_quat_batched(mats):
    q = roma.rotmat_to_unitquat(mats)
    q = roma.quat_xyzw_to_wxyz(q)
    return q
