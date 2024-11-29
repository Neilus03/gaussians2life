from typing import List
import numpy as np
from typing import Tuple


def gaussian_sample_from_anchor_point(
    anchor_point: List[float],
    std_elevation: float,
    std_azimuth: float,
    std_distance: float,
    num_samples: int = 1,
    lookat_center=None):
    """
    Samples points with the same distance from lookat_center (default: [0,0,0]) from a Gaussian distribution centered
    at the anchor point.

    :param anchor_point: Anchor point coordinates (x,y,z)
    :param num_samples: Number of samples to generate
    :param std_elevation: Standard deviation of the elevation angle
    :param std_azimuth: Standard deviation of the azimuth angle
    :param std_distance: Standard deviation of the distance (as factor of the distance center-anchor)
    :param lookat_center: If not None, the anchor point is considered to be relative to this point
    :return: Sampled points (num_samples, 3)
    """
    anchor_point = np.array(anchor_point)
    if lookat_center is not None:
        lookat_center = np.array(lookat_center)
        anchor_point = anchor_point - lookat_center
    assert not (anchor_point[0] == 0 and anchor_point[1] == 0), "Anchor point cannot be on the z-axis for this function"

    r = np.linalg.norm(anchor_point)
    anchor_phi = np.arctan2(anchor_point[1], anchor_point[0])
    anchor_elevation = np.arccos(anchor_point[2] / r)

    azimuth = np.random.normal(anchor_phi, std_azimuth, num_samples)
    elevation = np.random.normal(anchor_elevation, std_elevation / 2, num_samples)
    r = np.random.normal(r, std_distance * r, num_samples)

    x = r * np.sin(elevation) * np.cos(azimuth)
    y = r * np.sin(elevation) * np.sin(azimuth)
    z = r * np.cos(elevation)

    pos = np.stack([x, y, z], axis=1)

    if lookat_center is not None:
        pos = pos + lookat_center

    return pos


def sample_once_viewpoints_from_anchor_point(
    anchor_point: List[float],
    azimuth_range: Tuple[float, float],
    elevation_range: Tuple[float, float],
    distance_range: Tuple[float, float],
    num_samples_per_direction: int = 10,
    std_azimuth: float = 0.1,
    std_elevation: float = 0.2,
    std_distance: float = 0.05,
    lookat_center: List[float] = None):
    """
    Samples two series of viewpoints going into opposite directions from the anchor point.

    :param anchor_point: Anchor point coordinates (x,y,z)
    :param azimuth_range: Fixed range of azimuth angles in degrees (in both directions)
    :param elevation_range: Range of elevation angles in degrees (uniformly sampled endpoint elevation from this range)
    :param distance_range: Range of distances relative from the anchor point (uniformly sampled endpoint distance from this range)
    :param num_samples_per_direction: Number of views to generate in total
    :param lookat_center: If not None, the anchor point is considered to be relative to this point
    :return: Sampled points in both directions (2, num_samples_per_direction, 3)
    """
    anchor_point = np.array(anchor_point)
    if lookat_center is not None:
        lookat_center = np.array(lookat_center)
        anchor_point = anchor_point - lookat_center

    azimuth_range = np.clip(azimuth_range, -180, 180) / 180 * np.pi
    elevation_range = np.clip(elevation_range, -90, 90) / 180 * np.pi

    r = np.linalg.norm(anchor_point)
    anchor_phi = np.arctan2(anchor_point[1], anchor_point[0])
    anchor_elevation = np.arccos(anchor_point[2] / r)

    pos1_elevation = np.random.uniform(anchor_elevation + elevation_range[0], anchor_elevation + elevation_range[1])
    pos1_distance = np.random.uniform(r + distance_range[0] * r, r + distance_range[1] * r)

    pos2_elevation = np.random.uniform(anchor_elevation + elevation_range[0], anchor_elevation + elevation_range[1])
    pos2_distance = np.random.uniform(r + distance_range[0] * r, r + distance_range[1] * r)

    pos1_azimuth = anchor_phi + azimuth_range[0]
    pos2_azimuth = anchor_phi + azimuth_range[1]

    pos1_direction_azimuth = np.linspace(anchor_phi, pos1_azimuth, num_samples_per_direction + 1)[1:]
    if std_azimuth > 0:
        pos1_direction_azimuth += np.random.normal(0, std_azimuth, num_samples_per_direction)
    pos1_direction_elevation = np.linspace(anchor_elevation, pos1_elevation, num_samples_per_direction + 1)[1:]
    if std_elevation > 0:
        pos1_direction_elevation += np.random.normal(0, std_elevation / 2, num_samples_per_direction)
    pos1_direction_elevation = np.clip(pos1_direction_elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    pos1_direction_distance = np.linspace(r, pos1_distance, num_samples_per_direction + 1)[1:]
    if std_distance > 0:
        pos1_direction_distance += np.random.normal(0, std_distance, num_samples_per_direction)

    pos2_direction_azimuth = np.linspace(anchor_phi, pos2_azimuth, num_samples_per_direction + 1)[1:]
    if std_azimuth > 0:
        pos2_direction_azimuth += np.random.normal(0, std_azimuth, num_samples_per_direction)
    pos2_direction_elevation = np.linspace(anchor_elevation, pos2_elevation, num_samples_per_direction + 1)[1:]
    if std_elevation > 0:
        pos2_direction_elevation += np.random.normal(0, std_elevation / 2, num_samples_per_direction)
    pos2_direction_elevation = np.clip(pos2_direction_elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    pos2_direction_distance = np.linspace(r, pos2_distance, num_samples_per_direction + 1)[1:]
    if std_distance > 0:
        pos2_direction_distance += np.random.normal(0, std_distance, num_samples_per_direction)

    pos1_x = pos1_direction_distance * np.sin(pos1_direction_elevation) * np.cos(pos1_direction_azimuth)
    pos1_y = pos1_direction_distance * np.sin(pos1_direction_elevation) * np.sin(pos1_direction_azimuth)
    pos1_z = pos1_direction_distance * np.cos(pos1_direction_elevation)

    pos2_x = pos2_direction_distance * np.sin(pos2_direction_elevation) * np.cos(pos2_direction_azimuth)
    pos2_y = pos2_direction_distance * np.sin(pos2_direction_elevation) * np.sin(pos2_direction_azimuth)
    pos2_z = pos2_direction_distance * np.cos(pos2_direction_elevation)

    pos1 = np.stack([pos1_x, pos1_y, pos1_z], axis=1)
    pos2 = np.stack([pos2_x, pos2_y, pos2_z], axis=1)

    if lookat_center is not None:
        pos1 = pos1 + lookat_center
        pos2 = pos2 + lookat_center

    return np.stack([pos1, pos2], axis=0)
