import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from foodClassifier.data import get_validation_transforms
from foodClassifier.model import get_model
# ─── Configuration ─────────────────────────────────────────────────────────────

CLASSES = [
    "apple_pie", "baklava", "caesar_salad", "eggs_benedict",
    "frozen_yogurt", "grilled_salmon", "nachos", "pizza", "tacos", "waffles"
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best_resnet.pt")
IMAGE_SIZE = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = get_validation_transforms(IMAGE_SIZE)

# ─── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and release it during shutdown."""
    print(f"Loading model on {DEVICE}...")
    model = get_model("resnet", num_classes=len(CLASSES))
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE)
    model.eval()             # disable dropout — all neurons active for inference
    app.state.model = model  # store so all route functions can access it
    print("Model ready.")

    yield

    print("Shutting down.")
    del app.state.model


# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Food Classifier API",
    description="Upload a food image and get top-5 predictions with confidence scores.",
    version="1.0.0",
    lifespan=lifespan        # ← wires up the lifespan function
)

# ─── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Return the service status, active device, and supported classes."""
    return {
        "status":  "ok",
        "device":  str(DEVICE),
        "classes": CLASSES
    }

# ─── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Validate an uploaded image and return its top-five predictions."""

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Must be jpeg, png, or webp."
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = app.state.model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(5)

    predictions = [
        {
            "label":      CLASSES[idx.item()],
            "confidence": round(prob.item(), 4)
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "top":         predictions[0],
        "predictions": predictions
    }