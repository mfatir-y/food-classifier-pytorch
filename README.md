# Food Classifier Project Plan (Food-101 + CNN + Web App)

A learning-focused project plan: build a CNN from scratch in PyTorch, serve it with FastAPI, and consume it from a Next.js/React frontend with live camera support.

---

## Phase 0 — Setup

1. Download the Food-101 dataset (zip) and extract it locally.
2. Inspect the folder structure — find `images/`, `meta/classes.txt`, `meta/train.txt`, `meta/test.txt`.
3. Decide on scope: start with a subset of 10-20 classes instead of all 101. This makes training/debugging much faster while you learn.
4. Set up your Python environment (PyTorch, torchvision, matplotlib, etc.) — locally or in a Kaggle Notebook with GPU.

---

## Phase 1 — Data Pipeline

1. Write a script to filter the dataset down to your chosen classes, splitting into `train/` and `test/` folders (or use `meta/train.txt` and `meta/test.txt` directly).
2. Explore the data: open a few images, check sizes, check class balance.
3. Define `torchvision.transforms` for training (resize, random crop, flip, normalize) and validation (resize, center crop, normalize).
4. Load the data using `ImageFolder` + `DataLoader`. Check that a batch loads correctly and visualize a few images with their labels.

---

## Phase 2 — Build the CNN

1. Sketch your architecture on paper first: how many conv blocks, how channels grow (e.g. 32 → 64 → 128 → 256), where pooling happens.
2. Implement the model as an `nn.Module`:
   - Convolution + ReLU + BatchNorm + MaxPool blocks for feature extraction
   - Adaptive pooling + fully connected layers for classification
3. Do a "shape check" — pass a dummy batch through the model and print tensor shapes after each block to make sure dimensions make sense.
4. Print the total parameter count to get a feel for model size.

---

## Phase 3 — Training Loop

1. Choose a loss function (`CrossEntropyLoss`) and optimizer (`Adam` is a good default).
2. Write the training loop: forward pass → loss → backward pass → optimizer step.
3. Write a validation loop: run on the test set without gradients, compute accuracy.
4. Add logging — print loss/accuracy per epoch, or use a list to track history for plotting later.
5. Train for a small number of epochs first (2-3) just to confirm the loop runs end-to-end without errors.
6. Once stable, train longer and watch for overfitting (train accuracy rising, val accuracy plateauing/falling).

---

## Phase 4 — Evaluate & Improve

1. Plot training/validation loss and accuracy curves.
2. Look at a confusion matrix — which classes get confused with each other?
3. Try improvements one at a time:
   - Add more conv layers or increase channel sizes
   - Add data augmentation (rotation, color jitter)
   - Adjust learning rate or add a learning rate scheduler
   - Try increasing image resolution
4. Save your best model's weights (`torch.save`).

---

## Phase 5 — Inference Script

1. Write a standalone script that:
   - Loads your saved model
   - Loads a single image from disk
   - Applies the same preprocessing as validation
   - Runs a forward pass and applies `softmax` to get probabilities
   - Prints the top-5 predicted classes with confidence percentages
2. Test this on a few images you didn't train on (download random food photos from the web).

---

## Phase 6 — Backend API (FastAPI)

1. Set up a basic FastAPI app with a health-check endpoint.
2. Add a `/predict` endpoint that:
   - Accepts an uploaded image file
   - Runs your Phase 5 inference logic
   - Returns JSON with top-5 predictions and confidence scores
3. Load the model once at startup (not per-request) for performance.
4. Test the endpoint locally using a tool like the FastAPI docs UI (`/docs`) or `curl`/Postman with a sample image.
5. Handle edge cases: invalid file types, oversized images, no file provided.

---

## Phase 7 — Frontend (Next.js + React)

1. Scaffold a Next.js app with a simple page.
2. Build an image upload component (drag-and-drop or file picker) that sends the image to your `/predict` endpoint and displays results.
3. Build a webcam component:
   - Access the camera with `getUserMedia`
   - Display the live video feed
   - Add a "capture" button that grabs a frame and converts it to a blob
   - Send the captured blob to `/predict`
4. Display results nicely:
   - Show the top prediction prominently
   - Show a list/bar chart of top-5 classes with confidence percentages
   - Add a loading state while waiting for the API response

---

## Phase 8 — Polish & Deploy (Optional)

1. Add basic styling (Tailwind or your preferred approach).
2. Handle errors gracefully in the UI (camera permission denied, API errors, etc.).
3. Deploy the backend (e.g. Render/Railway) and frontend (e.g. Vercel).
4. Update frontend API URL to point to the deployed backend.
5. Test the full flow end-to-end on a deployed URL, including camera access (requires HTTPS).

---

## Stretch Goals (If You Want to Go Further)

- Compare your from-scratch CNN against a transfer-learning model (ResNet) on the same classes.
- Add a "training mode" page that shows live loss/accuracy graphs while training runs (via WebSocket).
- Visualize what the CNN "sees" using feature map visualizations or Grad-CAM.
- Expand to all 101 classes once your pipeline is solid.
- Add a history/log of past predictions stored client-side.