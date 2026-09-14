# Food Classifier PyTorch

A learning-focused project: build CNNs from scratch in PyTorch on a 10-class subset of Food-101, serve the best model with FastAPI, and consume it from a Next.js frontend with live camera support. See `README.md` for the full phased plan and `MODEL_CHOICES_SO_FAR.md` for detailed reasoning behind each choice below.

## Classes

10 of Food-101's 101 classes: `apple_pie, baklava, caesar_salad, eggs_benedict, frozen_yogurt, grilled_salmon, nachos, pizza, tacos, waffles` (defined in `foodClassifier/predict.py` and `foodClassifier/main.py`).

## Structure

- `foodClassifier/model.py` — `BasicCNN`, `DeepCNN`, `ResNetTransfer`, all behind `get_model(name, num_classes)`
- `foodClassifier/data.py` — `Food101Subset` dataset + `get_dataloaders()`
- `foodClassifier/train.py` — training/validation loop, checkpointing, plotting
- `foodClassifier/predict.py` — standalone inference (CLI + importable `predict()`)
- `foodClassifier/main.py` — FastAPI app (`/health`, `/predict`)
- `food-cnn.ipynb` — the working notebook: data → model checks → training runs → evaluation
- `food-classifier-web/` — Next.js/TS frontend; `lib/api.ts` is wired up, `app/page.tsx` UI not built yet
- `best_basic.pt`, `best_deep.pt`, `best_resnet.pt` — checkpoints per model (gitignored)

## Current state

Phases 0–6 (data, models, training, evaluation, inference script, FastAPI backend) are done. Phase 7 (frontend upload/webcam UI) is started (API client exists) but the page itself is still the default scaffold. Phase 8 (deploy) not started.

## Model choices and reasoning

### Three models, one shared interface
All models subclass `FoodClassifierBase` (`forward()` + `count_parameters()`) so the same training loop, evaluation code, and inference path work for any of them without special-casing.

- **BasicCNN** — 4 Conv-BatchNorm-ReLU-MaxPool blocks (channels 3→32→64→128→256) + adaptive avg pool + small FC head. A simple from-scratch baseline, chosen to make the effect of each layer easy to reason about before adding complexity.
- **DeepCNN** — adds a 5th block (256→512 channels) and a wider FC head (512→1024→classes) with slightly higher dropout to compensate for the extra capacity. Exists to answer a concrete question: does more from-scratch capacity actually help on this dataset, or does it just overfit?
- **ResNetTransfer** — pretrained ResNet-50 backbone (ImageNet weights) with the backbone **frozen** and a custom head replacing `resnet.fc`. Reuses image features that are already useful rather than relearning them from 10 classes' worth of data, and gives a stronger comparison point against the from-scratch models. `unfreeze_backbone(layers=N)` is provided for later fine-tuning at a lower LR, kept separate so initial training on the frozen backbone stays fast and stable first.

All three are exposed through `get_model(name)` so `train.py` and the notebook can request a model by string without importing every class.

### Training loop design
- **Separate `train_one_epoch()` / `validate()` functions**, not one combined function — lets the notebook show training output first and validation later, and keeps the "am I overfitting" comparison visually separate.
- **Device-aware**: model and every batch are moved to the same `device` (CUDA if available) right before the forward pass, specifically to avoid tensor-device-mismatch errors and let the same code run unmodified on different machines.
- **CrossEntropyLoss + Adam + ReduceLROnPlateau**: treated as strong, low-friction defaults for a first multi-class image project rather than something to tune early. The scheduler watches val accuracy and halves LR after 3 stagnant epochs.
- **Early stopping** on val accuracy plateau (`patience` epochs), to avoid wasting compute once a model has converged.
- **Checkpointing**: each model saves to its own `.pt` file (`best_basic.pt`, `best_deep.pt`, `best_resnet.pt`) via `state_dict()` only (not the whole model object), specifically so runs can be compared later and reloaded without retraining.

### Inference/serving design
- **`predict.py` loads the model once at import time**, not per-call — this is the same "load once" principle the FastAPI app follows, avoiding repeated checkpoint loads for both CLI and API usage.
- **`predict()` accepts either a file path or a PIL.Image** so the exact same function serves the CLI script and the API endpoint (which decodes an upload into a PIL image) — one inference path, no duplicated preprocessing logic.
- **Preprocessing is shared** between training validation and inference via `get_validation_transforms()` in `data.py`, so API/CLI results stay consistent with what was measured during training.
- **FastAPI app loads the model once during `lifespan` startup** (not per-request) and stores it on `app.state`. Upload validation (content-type allowlist, 10MB size cap) happens *before* image decoding to avoid unnecessary work on bad input. Images are forced to RGB on load so grayscale/RGBA uploads don't break the 3-channel model input. CORS is fully open (`*`) because this is local-dev-only for now, not a deployed service.
- **Top-k output**: softmax over logits, then `topk()`, mapped back to label strings — chosen so both the CLI and the API return the same readable `{label, confidence}` shape, which the frontend's `predictions` array in `lib/api.ts` consumes directly.

## Known gaps / inconsistencies

- `MODEL_CHOICES_SO_FAR.md` references a `foodClassifier/config.json` used by the notebook to configure `train()` calls — this file doesn't currently exist in the repo; check whether it was removed or never committed before relying on it.
- `food-classifier-web/app/page.tsx` is still the unmodified `create-next-app` scaffold; the actual upload/webcam UI (Phase 7) hasn't been built yet even though the API client (`lib/api.ts`) is ready for it.
