import sys
import torch
import torch.nn.functional as F
from PIL import Image

from data import get_validation_transforms
from model import get_model

CLASSES = [
    "apple_pie", "baklava", "caesar_salad", "eggs_benedict",
    "frozen_yogurt", "grilled_salmon", "nachos", "pizza", "tacos", "waffles"
]

MODEL_PATH  = "best_resnet.pt"
IMAGE_SIZE  = 224          # ResNet was trained on 224x224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


preprocess = get_validation_transforms(IMAGE_SIZE)


def load_model():
    model = get_model("resnet", num_classes=len(CLASSES))
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE)
    model.eval()
    return model

_model = load_model()


def predict(image_input, top_k: int = 5):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path string or a PIL Image")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(CLASSES)))

    results = [
        {
            "label":      CLASSES[idx.item()],
            "confidence": round(prob.item(), 4)
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py pizza.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\nAnalyzing: {image_path}")
    print(f"Device: {DEVICE}\n")

    predictions = predict(image_path)

    print("Top predictions:")
    print("─" * 35)
    for i, result in enumerate(predictions, 1):
        bar_length = int(result["confidence"] * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{i}. {result['label']:<20} {result['confidence']*100:5.1f}%  {bar}")