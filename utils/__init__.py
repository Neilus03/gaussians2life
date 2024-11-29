from .loss_utils import l1_loss, l2_loss, ssim, rigidity_loss, arap_loss, jsd_loss, FlowLoss, rotation_similarity_loss, \
    longterm_isometry_loss
from .data_utils import gaussian_sample_from_anchor_point, sample_once_viewpoints_from_anchor_point
from .warping import find_homography, transform_video_with_homography, warp_video_with_flow
from .flow_back_projection import FlowBackProjection
from .render_utils import build_intrinsics

DEBUG = False


def set_debug(value):
    global DEBUG
    DEBUG = value


def dprint(text):
    if DEBUG:
        print(text)
