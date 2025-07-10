# LTX_preprocessing.py
#
# This script demonstrates the exact preprocessing pipeline for an image
# rendered by Gaussians-to-Life before it is passed to the LTX-Video guidance module.
# Run this script from the root of the 'neilus03-gaussians2life' project.
#
# The pipeline is as follows:
# 1. Start with a rendered tensor (simulated here by loading an image).
# 2. Convert the tensor to a PIL Image.
# 3. Pass the PIL Image to the LTX preprocessing function, which then:
#    a. Performs an aspect-ratio crop (which has no effect here).
#    b. Skips resizing because `just_crop=True` is used.
#    c. Applies a Gaussian blur.
#    d. Converts to a tensor and normalizes it to the [-1, 1] range.
#    e. Expands dimensions to 5D for the video model.
#    f. Checks for and applies padding if needed (not needed for 512x320).

import torch
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import torchvision.transforms.functional as TF

# --- Helper Functions ---

def print_tensor_stats(tensor, name: str):
    """Prints statistics of a tensor, numpy array, or PIL Image."""
    print(f"--- Stats for: {name} ---")
    if isinstance(tensor, torch.Tensor):
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Min: {tensor.min().item():.4f}")
        print(f"  Max: {tensor.max().item():.4f}")
        print(f"  Mean: {tensor.mean().item():.4f}")
        print(f"  Std: {tensor.std().item():.4f}")
    elif isinstance(tensor, np.ndarray):
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Min: {tensor.min():.4f}")
        print(f"  Max: {tensor.max():.4f}")
        print(f"  Mean: {tensor.mean():.4f}")
        print(f"  Std: {tensor.std():.4f}")
    elif isinstance(tensor, Image.Image):
        print(f"  Type: PIL.Image.Image")
        print(f"  Size (W, H): {tensor.size}")
        print(f"  Mode: {tensor.mode}")
    else:
        print(f"  Unsupported type: {type(tensor)}")
    print("-" * (len(name) + 12))
    print()

def save_tensor_as_image(tensor, file_path: Path, name: str):
    """Saves a tensor as an image file, handling denormalization and permuting."""
    print(f"Saving '{name}' to {file_path}...")
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Squeeze batch and frame dimensions if they exist
    if tensor.dim() == 5:
        tensor = tensor.squeeze(2).squeeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Permute from (C, H, W) to (H, W, C) for image saving
    if tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)
        
    img_np = tensor.cpu().numpy()
    
    # Denormalize if in [-1, 1] range
    if img_np.min() >= -1.0001 and img_np.max() <= 1.0001:
        print("  (Denormalizing from [-1, 1] to [0, 255])")
        img_np = (img_np + 1.0) * 127.5
    # Scale if in [0, 1] range
    elif img_np.min() >= 0.0 and img_np.max() <= 1.0001:
        print("  (Scaling from [0, 1] to [0, 255])")
        img_np *= 255.0
        
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
        
    Image.fromarray(img_np).save(file_path)
    print("Done.\n")
    
def crf_compress(x: torch.Tensor) -> torch.Tensor:
    """Placeholder identity function as seen in the LTX codebase."""
    return x

def calculate_padding(source_h, source_w, target_h, target_w):
    pad_h = target_h - source_h
    pad_w = target_w - source_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)

def main():
    # --- Configuration ---
    # This simulates the starting conditions from `gaussians2life`.
    # We use a real image and resize it to mimic the rendered output.
    simulated_render_source_path = Path("./assets/sample_data/bear.ply") # Use a file to know what scene we are using
    # Using lego-up since we used it in the other file.
    # Note: `bear.ply` is a 3D model, we need an image. We'll use a placeholder.
    # Let's assume we have a rendered image of the bear scene. We'll use a placeholder image.
    # For a reproducible example, let's just create a dummy image. Or better, use a real one
    # and resize it to match the config. Let's use an image from the previous exercise.
    # A local copy would be better. Let's assume a file `placeholder_render.png` exists.
    try:
        from ltx_video import __file__ as ltx_pkg_file
        ltx_image_path = Path(ltx_pkg_file).parents[1] / 'images/lego-up.png'
    except (ImportError, FileNotFoundError):
        print("Could not find LTX-Video package to get placeholder image.")
        print("Please ensure a placeholder image 'placeholder_render.png' exists or adjust the path.")
        ltx_image_path = Path('placeholder_render.png')

    output_dir = Path("./G2L_LTX_preprocessing_steps")
    
    # Dimensions from `configs/bear_ltx.yaml`
    render_height = 320
    render_width = 512
    
    device = torch.device("cpu")
    
    print("Starting Gaussians-to-Life -> LTX preprocessing demonstration.")
    print(f"Simulating a render from a 3DGS scene.")
    print(f"Output directory for steps: {output_dir}")
    print(f"Render HxW from config: {render_height}x{render_width}")
    print("-" * 50 + "\n")

    # --- Step 0: Simulate the Rendered Input Tensor ---
    # In Gaussians-to-Life, this is `out["comp_rgb"]` from the renderer.
    # It's a float tensor with shape (H, W, C) and values in [0, 1].
    # We simulate it by loading an image and converting it to this format.
    try:
        source_pil = Image.open(ltx_image_path).convert("RGB").resize((render_width, render_height))
        rendered_tensor_hwc_01 = (torch.from_numpy(np.array(source_pil)).float() / 255.0).to(device)
    except FileNotFoundError:
        print(f"ERROR: Could not find placeholder image at {ltx_image_path}. Exiting.")
        return

    print_tensor_stats(rendered_tensor_hwc_01, "0_Simulated_Rendered_Tensor")
    print("  Note: This tensor represents the output of the 3DGS renderer.")
    save_tensor_as_image(rendered_tensor_hwc_01.clone(), output_dir / "0_simulated_render.png", "0_Simulated_Render")

    # --- Step 1: Permute for PIL Conversion ---
    # `ltx_video_guidance.py` permutes from (H, W, C) to (C, H, W) before creating a PIL image.
    rendered_tensor_chw_01 = rendered_tensor_hwc_01.permute(2, 0, 1)
    print_tensor_stats(rendered_tensor_chw_01, "1_Permuted_Tensor_for_PIL")

    # --- Step 2: Convert to PIL Image ---
    # This is a key step where the raw tensor data becomes a standard image object.
    # `TF.to_pil_image` handles the conversion from a [0, 1] float tensor to an 8-bit image.
    pil_image = TF.to_pil_image(rendered_tensor_chw_01.cpu())
    print_tensor_stats(pil_image, "2_Converted_to_PIL_Image")
    save_tensor_as_image(torch.from_numpy(np.array(pil_image)), output_dir / "2_pil_conversion.png", "2_PIL_Conversion")

    # --- The following steps replicate `load_image_to_tensor_with_resize_and_crop` from LTX ---
    
    # --- Step 3: Aspect Ratio Crop (with just_crop=True) ---
    # Because `just_crop=True` is used, the resize step inside the LTX function is SKIPPED.
    # The aspect ratio crop is still calculated.
    input_w, input_h = pil_image.size
    target_aspect = render_width / render_height
    frame_aspect = input_w / input_h

    if abs(target_aspect - frame_aspect) > 1e-4: # Check if crop is needed
        if frame_aspect > target_aspect:
            new_w = int(input_h * target_aspect)
            x_start = (input_w - new_w) // 2
            image_cropped_pil = pil_image.crop((x_start, 0, x_start + new_w, input_h))
        else:
            new_h = int(input_w / target_aspect)
            y_start = (input_h - new_h) // 2
            image_cropped_pil = pil_image.crop((0, y_start, input_w, y_start + new_h))
    else:
        print("Aspect ratio already matches target. Cropping will have no effect.")
        image_cropped_pil = pil_image

    print_tensor_stats(image_cropped_pil, "3_AspectRatio_Cropped_PIL")
    print("  Note: Since input was 512x320, aspect ratio matches, so no change.")
    save_tensor_as_image(torch.from_numpy(np.array(image_cropped_pil)), output_dir / "3_aspect_crop.png", "3_Aspect_Crop")

    # --- Step 4: Gaussian Blur ---
    # The LTX function applies a blur. Input is a PIL image, converted to NumPy for cv2.
    image_for_blur_np = np.array(image_cropped_pil)
    image_blurred_np = cv2.GaussianBlur(image_for_blur_np, (3, 3), 0)
    print_tensor_stats(image_blurred_np, "4_Blurred_Numpy")
    save_tensor_as_image(torch.from_numpy(image_blurred_np), output_dir / "4_blurred.png", "4_Blurred")

    # --- Step 5: To Tensor & "CRF" ---
    # The numpy array is converted to a torch tensor. Range is [0, 255].
    frame_tensor_255 = torch.from_numpy(image_blurred_np).float().to(device)
    tensor_for_crf = frame_tensor_255 / 255.0
    tensor_after_crf = crf_compress(tensor_for_crf)
    frame_tensor_255 = tensor_after_crf * 255.0
    print_tensor_stats(frame_tensor_255, "5_Tensor_HWC_0-255")

    # --- Step 6: Permute and Normalize to [-1, 1] ---
    frame_tensor_m11 = frame_tensor_255.permute(2, 0, 1)
    frame_tensor_m11 = (frame_tensor_m11 / 127.5) - 1.0
    print_tensor_stats(frame_tensor_m11, "6_Permuted_Normalized_Tensor_CHW")
    save_tensor_as_image(frame_tensor_m11.clone(), output_dir / "6_normalized.png", "6_Normalized")

    # --- Step 7: Add Batch and Frame Dimensions ---
    final_tensor_5d = frame_tensor_m11.unsqueeze(0).unsqueeze(2)
    print_tensor_stats(final_tensor_5d, "7_Final_5D_Tensor_for_Conditioning")

    # --- Step 8: Padding Check (as done in LTX pipeline) ---
    # The LTX pipeline itself will check for padding. Let's replicate that check.
    # The target dimensions for LTX are taken from the guidance config (same as render dims).
    ltx_h, ltx_w = render_height, render_width
    h_padded_target = ((ltx_h - 1) // 32 + 1) * 32
    w_padded_target = ((ltx_w - 1) // 32 + 1) * 32
    
    print(f"Padding check: Target HxW={ltx_h}x{ltx_w}, Padded Target HxW={h_padded_target}x{w_padded_target}")
    
    padding_values = calculate_padding(ltx_h, ltx_w, h_padded_target, w_padded_target)
    
    if any(p > 0 for p in padding_values):
        print(f"Applying padding: (L, R, T, B) = {padding_values}")
        final_tensor_padded = F.pad(final_tensor_5d, padding_values)
    else:
        print("No padding required as dimensions are divisible by 32.")
        final_tensor_padded = final_tensor_5d

    print_tensor_stats(final_tensor_padded, "8_Final_Padded_Tensor_for_LTX")
    print("This is the final tensor used as the conditioning item for LTX-Video.")
    save_tensor_as_image(final_tensor_padded.clone(), output_dir / "8_final_preprocessed.png", "8_Final_Preprocessed")

if __name__ == "__main__":
    main()