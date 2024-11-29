import cv2
import numpy as np
import torch

flow_model = None


def warp_with_flow(img, flow):
    if img.shape[-1] > 4:
        img = img.transpose(1, 2, 0)
        transpose = True
    else:
        transpose = False
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = x - flow[0]
    y_new = y - flow[1]
    res = cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE)
    if transpose:
        return res.transpose(2, 0, 1)
    else:
        return res


def warp_video_with_flow(static_img_1, static_img_2, vid1, static_index_1, flow_model_size="large"):
    device = vid1.device

    static_img_1 = torch.tensor(static_img_1).float().to(device).permute(2, 0, 1)
    static_img_2 = torch.tensor(static_img_2).float().to(device).permute(2, 0, 1)

    # Initialize the optical flow model in case it is not yet initialized
    global flow_model
    if flow_model is None:
        if flow_model_size == "small":
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device)
        elif flow_model_size == "large":
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
        else:
            raise ValueError("Invalid flow model size")

    # Compute the optical flow between the two static images and the optical flow between the frames of the first video
    static_img_1 = 2 * static_img_1 - 1
    static_img_2 = 2 * static_img_2 - 1
    vid1_in = 2 * vid1 - 1

    with torch.no_grad():
        flow_1_2 = flow_model(static_img_1[None], static_img_2[None])[-1][0]
        # Compute the optical flow from other frames to the static frame
        # if static_index_1 != 0:
        #     flow_vid1.append(flow_model(vid1_in[:static_index_1], vid1_in[1:static_index_1 + 1])[-1].cpu())
        # if static_index_1 != len(vid1_in) - 1:
        #     flow_vid1.append(torch.flip(
        #         flow_model(torch.flip(vid1_in[static_index_1 + 1:], [0]), torch.flip(vid1_in[static_index_1:-1], [0]))[
        #             -1].cpu(), [0]))
        # flow_vid1 = torch.cat(flow_vid1)

        flow_vid1 = flow_model(torch.cat([vid1_in[:static_index_1], vid1_in[static_index_1 + 1:]]), vid1_in[static_index_1][None].repeat(len(vid1_in) - 1, 1, 1, 1))[-1]

    # Warp the frames of the first video to the second video
    num_frames, H, W, _ = vid1.shape
    vid1 = vid1.cpu().numpy()
    warped_vid2 = [None] * num_frames
    flow_1_2 = flow_1_2.cpu().numpy()
    composed_flow = None
    for i in range(static_index_1 - 1, -1, -1):
        next_flow = flow_vid1[i].cpu().numpy()

        if composed_flow is None:
            composed_flow = next_flow
        else:
            warped_flow = warp_with_flow(composed_flow, next_flow)
            composed_flow += warped_flow

        flow_to_vid2 = warp_with_flow(flow_1_2, composed_flow)

        # warp the i-th frame from the first video to the i-th frame of the second video
        warped_vid2[i] = warp_with_flow(vid1[i], flow_to_vid2)
        composed_flow = None

    warped_vid2[static_index_1] = warp_with_flow(vid1[static_index_1], flow_1_2)

    composed_flow = None
    for i in range(static_index_1, num_frames - 1):
        next_flow = flow_vid1[i].cpu().numpy()

        if composed_flow is None:
            composed_flow = next_flow
        else:
            warped_flow = warp_with_flow(composed_flow, next_flow)
            composed_flow += warped_flow

        flow_to_vid2 = warp_with_flow(flow_1_2, composed_flow)

        # warp the i-th frame from the first video to the i-th frame of the second video
        warped_vid2[i + 1] = warp_with_flow(vid1[i + 1], flow_to_vid2)
        composed_flow = None

    return torch.tensor(np.stack(warped_vid2, axis=0)).clamp(0, 1).to(device) #.permute(0, 3, 1, 2)


##############
# DEPRECATED #
##############
# This code was used to warp the video from the previous viewpoint to the current viewpoint in the first version of the
# paper. We found that using an optical flow-based warping method was more effective, so this code is no longer used.

def find_homography(img1, img2):
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match the descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Compute the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def transform_video_with_homography(video, H):
    device = video.device
    video = video.cpu().numpy()
    video = np.transpose(video, (0, 2, 3, 1))
    out = np.zeros_like(video)
    height, width, channels = video[0].shape
    for i in range(len(video)):
        out[i] = cv2.warpPerspective(video[i], H, (width, height), borderMode=cv2.BORDER_REPLICATE)
    out = np.transpose(out, (0, 3, 1, 2))
    return torch.tensor(out, device=device)
