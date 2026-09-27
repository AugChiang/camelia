"""SMP segmentation CLI; keep source images, hard masks, and confidence previews."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camelia.segmentation import arguments, run


def load_predictor(args):
    import cv2
    import numpy as np
    import torch
    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    checkpoint = args.checkpoint or Path(__file__).resolve().parent / "pretrained" / f"best_{args.model_type}_model.pth"
    # Load on CPU first rather than creating a second copy of checkpoint weights on CUDA.
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(state)
    del state
    model.to(device).eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.inference_mode()
    def predict(image):
        # Preserve the pretrained model's grayscale, nearest-resize, ImageNet normalization.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        normalized = (np.repeat(resized[..., None], 3, axis=2).astype(np.float32) / 255 - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0).to(device)
        probabilities = model(tensor).sigmoid()[0, 0].cpu().numpy()
        return cv2.resize(probabilities, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return predict


if __name__ == "__main__":
    sys.exit(run(arguments("smp", ROOT), "smp", load_predictor))
