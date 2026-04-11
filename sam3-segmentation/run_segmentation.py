import os
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import tempfile
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OUTPUT_DIR = "output/"

def load_sam3_model(model_path):
    """Load the SAM3 model and processor."""
    # load model and processor
    model = build_sam3_image_model(checkpoint_path=model_path, device=DEVICE)
    processor = Sam3Processor(model, confidence_threshold=0.5, device=DEVICE)
    return processor

def get_label_prompt(model_type):
    """Get the label prompt based on the model type."""
    if model_type in ["black_bars", "transparent_black"]:
        return "black bars, black rectangles, black stripes"
    elif model_type == "white_bars":
        return "white bars, white rectangles, white stripes"
    else:
        raise ValueError("Invalid model type. Please choose 'black_bars', 'white_bars', or 'transparent_black'.")

def predict_mask_sam3(processor, image_path, label_prompt):
    """Generate a mask prediction using SAM3."""
    img_pil = Image.open(image_path)
    
    # align to bfloat16 for faster inference if on CUDA
    if DEVICE == "cuda":
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(img_pil)
            processor.reset_all_prompts(inference_state)
            output = processor.set_text_prompt(
                state=inference_state, 
                prompt=label_prompt
            )
    else:
        with torch.no_grad():
            inference_state = processor.set_image(img_pil)
            processor.reset_all_prompts(inference_state)
            output = processor.set_text_prompt(
                state=inference_state, 
                prompt=label_prompt
            )
    
    output_masks = output["masks"] # shape=[num_masks,1,H,W]
    if output_masks.shape[0] == 0:
        # Return a zero mask if no bars were detected
        w, h = img_pil.size
        return np.zeros((h, w), dtype=np.float32)
    
    # Combine masks by taking the maximum across all detected masks
    combined_mask = torch.max(output_masks, dim=0).values.squeeze().detach().cpu().numpy()
    return combined_mask

def create_opacity_mask(predicted_mask):
    """
    Create an opacity mask based on thresholds.
    SAM3 masks are usually boolean or 0/1. If it's 0/1, we'll map to 255 (full opacity)
    or use thresholds if they are probabilities.
    Assuming SAM3 output masks can be probabilities if thresholds aren't applied inside set_text_prompt.
    Based on smp script, we'll use similar thresholds.
    """
    opacity_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
    opacity_mask[predicted_mask > 0.2] = 50
    opacity_mask[predicted_mask > 0.35] = 75
    opacity_mask[predicted_mask > 0.5] = 100
    return opacity_mask

def save_results(original_image, predicted_mask, output_path, relative_output_path):
    """Save the original image and predicted mask to the output directory."""
    images_output_dir = os.path.join(output_path, "images", os.path.dirname(relative_output_path))
    masks_output_dir = os.path.join(output_path, "masks", os.path.dirname(relative_output_path))

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    if predicted_mask.max() > 0:
        predicted_mask = (predicted_mask / predicted_mask.max() * 255).astype(np.uint8)

    h, w = original_image.shape[:2]
    resized_mask = cv2.resize(predicted_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    original_image_path = os.path.join(images_output_dir, os.path.basename(relative_output_path))
    Image.fromarray(original_image).save(original_image_path, format="PNG")

    mask_path = os.path.join(masks_output_dir, os.path.basename(relative_output_path))
    Image.fromarray(resized_mask).save(mask_path, format="PNG")

def convert_to_png(image_path):
    """Convert JPEG or WebP image to PNG format if needed."""
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.png':
        return image_path, False
    
    try:
        workspace_root = os.environ.get('WORKSPACE_ROOT', '')
        display_path = image_path
        if workspace_root and image_path.startswith(workspace_root):
            display_path = image_path[len(workspace_root):].lstrip(os.sep)
        
        print(f"Converting {ext} image to PNG: {display_path}")
        img = Image.open(image_path)
        
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        
        img.save(temp_path, format='PNG')
        return temp_path, True
    except Exception as e:
        print(f"Error converting image {display_path}: {e}")
        return image_path, False

def process_directory_recursively(input_dir, output_dir, processor, model_type):
    temp_files = []
    workspace_root = os.environ.get('WORKSPACE_ROOT', '')
    label_prompt = get_label_prompt(model_type)
    
    try:
        for root, _, files in os.walk(input_dir):
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(root, file)
                    try:
                        # Convert to PNG if needed
                        png_path, is_temp = convert_to_png(image_path)
                        if is_temp:
                            temp_files.append(png_path)
                        
                        display_path = image_path
                        if workspace_root and image_path.startswith(workspace_root):
                            display_path = image_path[len(workspace_root):].lstrip(os.sep)
                        
                        print(f"Processing (SAM3): {display_path}")
                        
                        # Use PIL to read original image for consistency
                        original_image = np.array(Image.open(png_path).convert("RGB"))
                        
                        predicted_mask = predict_mask_sam3(processor, png_path, label_prompt)
                        if predicted_mask.shape[0] == 0 or predicted_mask.shape[1] == 0:
                            print(f"No bars detected in {display_path}, skipping mask saving.")
                            continue
                        opacity_mask = create_opacity_mask(predicted_mask)
                        
                        output_filename = os.path.splitext(file)[0] + '.png'
                        save_results(original_image, opacity_mask, output_dir, 
                                    os.path.join(relative_path, output_filename))
                    except Exception as e:
                        print(f"Error processing {display_path}: {e}")
    finally:
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

def run_inference(model_path, model_type, base_input_dir, output_dir):
    """Run SAM3 inference on all images in the input directory."""
    processor = load_sam3_model(model_path)
    
    # In this script, input_dir passed is already the model_type subfolder OR the base folder?
    # In main.py:
    # camelia_input = os.path.join(workspace_root, "camelia-decensor", "input")
    # input_dir=camelia_input
    # But SMP script does os.path.join(base_input_dir, model_type).
    # So I'll do the same.
    input_dir = os.path.join(base_input_dir, model_type)

    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"Created input directory: {input_dir}")
        print(f"Please place input images in {input_dir}")
        return

    process_directory_recursively(input_dir, output_dir, processor, model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation inference.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["black_bars", "white_bars", "transparent_black"],
        help="Specify the model type to use."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="input",
        help="Base input directory (model_type subfolder will be appended)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for segmentation results"
    )
    args = parser.parse_args()

    workspace_root = os.environ.get('WORKSPACE_ROOT', '.')
    model_path = os.path.join(workspace_root, "sam3", "pretrained", "sam3.pt")
    
    run_inference(model_path, args.model_type, args.input_dir, args.output_dir)
