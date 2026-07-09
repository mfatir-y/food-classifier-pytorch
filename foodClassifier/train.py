import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ─── Training pass (one epoch) ────────────────────────────────────────────────
# Returns average loss and accuracy for the epoch.

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()   # enables dropout + batchnorm training mode
    
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        # Move data to correct device (GPU or CPU), model is already there
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)              # forward pass → [B, num_classes] raw scores
        loss    = criterion(outputs, labels) # compare scores to ground truth
        loss.backward()                      # backprop — compute gradients for every weight
        optimizer.step()                     # update weights using those gradients

        # Track metrics
        total_loss += loss.item() * images.size(0)   # loss.item() is per-batch average,
                                                     # multiply back to get total for the batch
        _, predicted = outputs.max(dim=1)            # index of highest score = predicted class
        correct      += (predicted == labels).sum().item()
        total        += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─── Validation pass (one epoch) ──────────────────────────────────────────────
# Same structure as training but:
#   - model.eval() disables dropout so all neurons are active
#   - torch.no_grad() skips building the computation graph (faster + less memory)
#   - no optimizer.zero_grad() / loss.backward() / optimizer.step()

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(dim=1)
            correct      += (predicted == labels).sum().item()
            total        += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─── Plot training curves ─────────────────────────────────────────────────────
# Two subplots side by side: loss on the left, accuracy on the right to check overfitting.

def plot_history(history, model_name="CNN"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title("Loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train accuracy")
    ax2.plot(epochs, history["val_acc"],   label="Val accuracy")
    ax2.set_title("Accuracy per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f"Training history — {model_name} CNN", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ─── Main training loop ───────────────────────────────────────────────────────

def train(
    train_loader=None,
    val_loader=None,
    model=None,
    epochs=20,
    lr=1e-3,
    patience=5,
    save_path="best_model.pt",
    model_name="basic",
    device=None,
):
    if model is None or train_loader is None or val_loader is None:
        raise ValueError("train_loader, val_loader, and model must all be provided.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # CrossEntropyLoss combines LogSoftmax + NLLLoss in one step.
    # It expects raw scores (logits) from the model, not softmax outputs.
    criterion = nn.CrossEntropyLoss()

    # Adam adapts the learning rate per parameter automatically.
    # Much more forgiving than plain SGD for a first project.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ReduceLROnPlateau watches val accuracy and divides LR by 2 (factor=0.5)
    # if it hasn't improved for 3 epochs. Helps escape plateaus late in training.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # History dict stores metrics every epoch for plotting afterward
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Early stopping state — stop training if val accuracy hasn't improved
    # for PATIENCE epochs. Saves time and prevents overfitting.
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        # Step scheduler — passes val accuracy so it knows whether to reduce LR
        scheduler.step(val_acc)

        # Log to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Current learning rate (may have been reduced by scheduler)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.1f}% | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save model if this is the best val accuracy so far.
        # Saving state_dict (weights only) rather than the whole model
        # is the standard — it's smaller and not tied to your class definition.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val acc: {val_acc:.1f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping — val accuracy hasn't improved for {patience} epochs.")
                break

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    plot_history(history, model_name=model_name)


if __name__ == "__main__":
    train()