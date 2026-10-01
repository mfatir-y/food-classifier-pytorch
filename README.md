# Food Classifier — Food-101 CNN + FastAPI + Next.js

A learning-focused project: train CNNs from scratch and via transfer learning in PyTorch on a 10-class subset of Food-101, serve the best model with a FastAPI backend, and classify food photos through a Next.js web app — upload or drag-and-drop an image, hit Classify, and get the top-5 predicted classes with confidence scores. Recent predictions are kept in a local history, stored in the browser.

## Live demo

Try the deployed app: <a href="https://food-classifier-pytorch.vercel.app/">Food Classifier</a>

## Classes

The project trains on a 10-class subset of Food-101 (not all 101 classes, to keep training/debugging fast):

`apple_pie` · `baklava` · `caesar_salad` · `eggs_benedict` · `frozen_yogurt` · `grilled_salmon` · `nachos` · `pizza` · `tacos` · `waffles`

## Project structure

```
foodClassifier/          Python package — models, data, training, inference, API
  model.py                 BasicCNN, DeepCNN, ResNetTransfer + get_model() factory
  data.py                  Food101Subset dataset, train/val transforms, get_dataloaders()
  train.py                 Training/validation loop, checkpointing, history plots
  predict.py               Standalone inference (CLI + importable predict())
  main.py                  FastAPI app — /health and /predict

food-cnn.ipynb            Working notebook: data → model checks → training → evaluation

food-classifier-web/      Next.js (App Router) + TypeScript frontend
  app/page.tsx               Page state/orchestration
  app/components/            UploadZone, ResultsPanel, HistoryPanel
  lib/                       api.ts (backend client), validation.ts, format.ts, history.ts

best_basic.pt              Saved checkpoints (gitignored — trained locally, not in the repo)
best_deep.pt
best_resnet.pt

test_images/               Sample images for manual testing
```

## Models

All three share a common interface (`FoodClassifierBase`) so the same training loop, evaluation code, and inference path work for any of them, selected by name via `get_model(name)`.

| Model | Architecture | Why |
|---|---|---|
| **BasicCNN** | 4× Conv‑BatchNorm‑ReLU‑MaxPool blocks (3→32→64→128→256 channels) + adaptive avg pool + small FC head | A simple from-scratch baseline, chosen to make the effect of each layer easy to reason about before adding complexity. |
| **DeepCNN** | Adds a 5th block (256→512 channels) and a wider FC head (512→1024→classes), with higher dropout | Tests whether more from-scratch capacity actually helps on this dataset, or just overfits. |
| **ResNetTransfer** | Pretrained ResNet-50 (ImageNet weights), backbone **frozen**, with a custom FC head replacing `resnet.fc`; `unfreeze_backbone(layers=N)` is available for later fine-tuning | Reuses image features that are already useful instead of relearning them from only 10 classes' worth of data — the strongest of the three, and the one actually served by the API. |

Each model saves its best checkpoint separately (`best_basic.pt`, `best_deep.pt`, `best_resnet.pt`) via `state_dict()`, so runs can be compared and reloaded without retraining. (`best_model.pt` is a leftover from an early run before per-model save paths were wired up — not part of the active set.)

Training uses `CrossEntropyLoss` + `Adam`, with `ReduceLROnPlateau` halving the learning rate after 3 stagnant epochs on validation accuracy, and early stopping once accuracy plateaus for `patience` epochs. BasicCNN/DeepCNN train at 128×128; ResNetTransfer trains at 224×224 (matching its ImageNet pretraining) and is what `predict.py`/`main.py` load for inference.

## Backend (FastAPI)

`foodClassifier/main.py` loads `best_resnet.pt` once at startup (via the app's `lifespan`) and serves:

- `GET /health` — status, active device (`cuda`/`cpu`), and the class list
- `POST /predict` — accepts an uploaded image (`multipart/form-data`), returns the top prediction plus the full top-5 list with confidences

Upload validation happens before decoding: only `image/jpeg`, `image/png`, and `image/webp` are accepted, and files over 10MB are rejected. Images are converted to RGB so grayscale/transparent uploads don't break the 3-channel input. CORS is fully open (`*`) since this is local-dev only for now.

Inference (`foodClassifier/predict.py`) applies the same preprocessing used for validation during training (`get_validation_transforms()` in `data.py`), so results stay consistent with what training measured. It also has a CLI entry point:

```bash
python -m foodClassifier.predict path/to/image.jpg
```

## Frontend (Next.js)

A single-page app (`food-classifier-web/`) built with Next.js 16 (App Router), React 19, TypeScript, and Tailwind CSS:

- **Upload** — click-to-browse or drag-and-drop, with client-side validation mirroring the backend's type/size rules
- **Classify button** — explicit submit (not auto-classify), with a loading state
- **Results** — the top prediction prominently, plus a top-5 confidence bar list
- **History** — the 10 most recent predictions (small thumbnail, label, confidence), persisted in the browser's `localStorage` via a `useSyncExternalStore`-backed store, with a Clear option

Component layout:
```
app/page.tsx              State + orchestration, plus small inline UI (classify button, error banner)
app/components/
  UploadZone.tsx             Drag/drop + file input + image preview
  ResultsPanel.tsx           Top prediction card + top-5 bars
  HistoryPanel.tsx           Recent predictions strip
lib/
  api.ts                      classifyImage(), checkHealth() — talks to the FastAPI backend
  validation.ts                Shared file-type/size validation
  format.ts                    Label formatting (apple_pie -> Apple Pie)
  history.ts                   localStorage-backed prediction history store
```

## Getting started

### Prerequisites

- Python 3.11+ with `pip`
- Node.js 20.9+ (required by Next.js 16) and `npm`
- A CUDA-capable GPU is optional but strongly recommended for training (inference runs fine on CPU)

### 1. Get the dataset

Download the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101) and extract it so you have a folder containing `images/` and `meta/` (with `classes.txt`, `train.txt`, `test.txt`).

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

For a CUDA build of PyTorch instead of the default CPU wheels, see the note at the top of `requirements.txt` (or [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)).

### 3. Train (or skip this if you already have checkpoints)

Open `food-cnn.ipynb`, set `DATA_PATH` in the first cell to your extracted Food-101 folder, and run the cells top to bottom. Each model's best weights are saved automatically (`best_basic.pt`, `best_deep.pt`, `best_resnet.pt`) as training runs.

### 4. Run the backend

```bash
uvicorn foodClassifier.main:app --reload
```
Runs at `http://127.0.0.1:8000` by default; requires `best_resnet.pt` to be present at the project root.

### 5. Run the frontend

```bash
cd food-classifier-web
npm install
npm run dev
```
Runs at `http://localhost:3000`. It talks to the backend at `NEXT_PUBLIC_API_URL` (defaults to `http://127.0.0.1:8000` — set it in `.env.local` if your backend runs elsewhere).

## Status & roadmap

Done: data pipeline, all three models, training/evaluation, standalone inference (CLI), FastAPI backend, and a working frontend (upload, classify, results, history).

Not yet done:
- Webcam capture (live camera feed + frame capture) alongside file upload
- Deployment (backend to Render/Railway-style hosting, frontend to Vercel)
- Feature-map/Grad-CAM visualizations of what the CNN "sees"
- Expanding beyond the current 10-class subset

## Known gaps

- `MODEL_CHOICES_SO_FAR.md` mentions a `foodClassifier/config.json` used by the notebook to configure `train()` calls — this file doesn't currently exist in the repo.

See `MODEL_CHOICES_SO_FAR.md` for a more detailed running log of what was added, how, and why for each model/training/serving decision.
