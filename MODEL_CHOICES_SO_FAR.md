# Model Choices So Far

This file is a working reference for the model-related choices that already exist in the project.
It is meant to support later writing about the project by showing what was added, how it was added, and why it was added.

## What Has Been Added So Far

- A Basic CNN baseline.
- A deeper CNN variant.
- A ResNet transfer-learning option.
- Training code that handles device placement, history tracking, checkpoint saving, and plotting.
- Notebook cells that test each model and load saved `.pt` weights for later comparison.

## Model Additions

### Basic CNN

What was added:
- A small convolutional baseline for food classification.

How it was added:
- Implemented as `BasicCNN` in `foodClassifier/model.py`.
- Uses four Conv-BatchNorm-ReLU-MaxPool blocks.
- Uses adaptive average pooling and a small fully connected head.
- Exposed through `get_model("basic", num_classes=10)`.
- Tested in the notebook with a dummy input tensor and parameter-count printout.

Why it was added:
- Gives a simple from-scratch baseline.
- Makes it easier to understand the effect of each layer.
- Provides a reference model before trying larger architectures.

### Deep CNN

What was added:
- A deeper version of the baseline CNN.

How it was added:
- Implemented as `DeepCNN` in `foodClassifier/model.py`.
- Extends the feature extractor to five convolution blocks.
- Uses a wider classifier head with more hidden units.
- Exposed through `get_model("deep", num_classes=10)`.
- Tested in the notebook the same way as the basic model.

Why it was added:
- Lets the project compare a deeper from-scratch model against the baseline.
- Helps show whether extra capacity improves validation performance.
- Gives a second architecture that still matches the same training pipeline.

### ResNet Transfer Model

What was added:
- A transfer-learning model based on pretrained ResNet-50 weights.

How it was added:
- Implemented as `ResNetTransfer` in `foodClassifier/model.py`.
- Loads pretrained ImageNet weights.
- Freezes the backbone parameters first.
- Replaces the final classifier with a custom head for the food classes.
- Adds an `unfreeze_backbone()` helper for later fine-tuning.
- Exposed through `get_model("resnet", num_classes=10)`.
- Tested in the notebook with a larger `224x224` input size.

Why it was added:
- Gives a stronger comparison point than training only from scratch.
- Reuses visual features that are already useful for image classification.
- Makes it possible to fine-tune later instead of retraining everything from zero.

## Training And Evaluation Choices

### Separate training and validation flow

What was added:
- Training and validation are split into separate functions.

How it was added:
- `train()` handles only the training loop.
- `run_validation()` handles validation later in the notebook.
- Separate plot functions were added for training history and validation history.

Why it was added:
- Lets the notebook show training output first and validation output later.
- Makes the project easier to explain step by step.
- Keeps the training slice and testing slice visually separate.

### Device-aware execution

What was added:
- Automatic CPU/GPU placement for model and batches.

How it was added:
- The training code selects CUDA when available.
- The model and each batch are moved to the same device before the forward pass.

Why it was added:
- Prevents tensor device mismatch errors.
- Lets the same code run on different machines without editing.

### Loss, optimizer, and scheduler

What was added:
- CrossEntropyLoss, Adam, and ReduceLROnPlateau.

How it was added:
- `nn.CrossEntropyLoss()` is used on raw logits.
- `torch.optim.Adam()` updates the weights.
- `ReduceLROnPlateau` lowers the learning rate when progress stalls.

Why it was added:
- These are strong defaults for a first multi-class image project.
- They make training easier to stabilize and easier to iterate on.

### Checkpoint saving

What was added:
- Best-model saving to separate `.pt` files.

How it was added:
- The notebook uses different save names such as `best_basic.pt`, `best_deep.pt`, and `best_resnet.pt`.
- Saved weights are restored later with `torch.load()` and `load_state_dict()`.

Why it was added:
- Keeps each model’s best weights separate.
- Makes it easy to compare runs later.
- Lets the notebook reload the best version without retraining.

## Older Facts Worth Preserving

These are smaller implementation details that are still worth remembering when describing the project later.

### Config values moved into the notebook

What was added:
- Notebook-side config loading from `foodClassifier/config.json`.

How it was added:
- The notebook reads the JSON file with `Path` and `json.load()`.
- The values are then passed into `train()` as arguments instead of being read inside `train.py`.

Why it was added:
- Keeps the training module focused on training logic.
- Makes notebook runs easier to customize per model.
- Preserves the project plan in the notebook while keeping the code modular.

### Training loop conventions

What was added:
- The usual PyTorch training pattern: `model.train()`, `optimizer.zero_grad()`, forward pass, loss, backward pass, optimizer step.

How it was added:
- Written into `train_one_epoch()`.
- Batches are moved to the active device before the forward pass.

Why it was added:
- Prevents gradient accumulation from previous batches.
- Makes the training behavior explicit and easy to explain.

### Validation conventions

What was added:
- A separate validation pass with `model.eval()` and `torch.no_grad()`.

How it was added:
- Written into `validate()`.
- Keeps inference-only behavior isolated from the training loop.

Why it was added:
- Makes validation deterministic.
- Reduces memory use and speeds up testing.

### Output for later comparison

What was added:
- Separate training and validation plots and separate `.pt` checkpoints per model.

How it was added:
- `plot_train_history()` and `plot_validation_history()` show the two phases separately.
- Each model saves its own checkpoint name, such as `best_basic.pt` or `best_resnet.pt`.

Why it was added:
- Makes it easier to compare models later in writing.
- Keeps the best learned weights available for reuse.

## What The Notebook Currently Shows

- A data-loading section for the Food-101 subset.
- A basic model check with shape and parameter count output.
- A deep model check with the same style of output.
- A ResNet section using a larger input size.
- A later cell that loads a saved `.pt` checkpoint and restores the model weights.

## Good Short Summary For Later Writing

The project started with a simple CNN baseline, then added a deeper CNN and a ResNet transfer-learning option for comparison. Each model was added directly in `foodClassifier/model.py`, exposed through `get_model()`, and tested in the notebook with shape checks and parameter counts. The training code was written to handle device placement, optimizer setup, checkpoint saving, and later validation so each model could be compared fairly and its best weights could be saved and reloaded from `.pt` files.
