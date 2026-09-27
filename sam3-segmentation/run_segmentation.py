"""SAM3 segmentation CLI with calibrated probability previews and hard masks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camelia.segmentation import arguments, run


def load_predictor(args):
    from contextlib import nullcontext
    import numpy as np
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    checkpoint = args.checkpoint or ROOT / "sam3/pretrained/sam3.pt"
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), device=str(device))
    processor = Sam3Processor(model, confidence_threshold=0.5, device=str(device))
    prompt = "white bars, white rectangles, white stripes" if args.model_type == "white_bars" else "black bars, black rectangles, black stripes"
    # Older CUDA devices may not support BF16. FP32 remains the safe fallback.
    use_bf16 = False
    if device.type == "cuda":
        with torch.cuda.device(device):
            use_bf16 = torch.cuda.is_bf16_supported()

    @torch.inference_mode()
    def predict(image):
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with context:
            state = processor.set_image(Image.fromarray(image))
            output = processor.set_text_prompt(state=state, prompt=prompt)
        # Despite its name, this processor's masks_logits field contains sigmoid probabilities.
        masks = output["masks_logits"]
        if masks.shape[0] == 0:
            return np.zeros(image.shape[:2], dtype=np.float32)
        return masks.amax(dim=0)[0].float().cpu().numpy()
    return predict


if __name__ == "__main__":
    sys.exit(run(arguments("sam3", ROOT), "sam3", load_predictor))
