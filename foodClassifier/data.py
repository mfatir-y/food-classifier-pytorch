import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class Food101Subset(Dataset):
    def __init__(self, data_path, classes, split="train", transform=None):
        # Store the provided image transformation parameter
        self.transform = transform
        # Store the mapping from class name to index from the original dataset for our classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)} 
        # Store the samples (image paths and labels) for the specified split
        self.samples = []

        # Read the text file corresponding to the specified split (train or test)
        txt_file = os.path.join(data_path, "meta", f"{split}.txt")
        with open(txt_file) as f:
            lines = f.read().splitlines()

        # Create the list of samples (image paths and class labels) 
        for line in lines:
            class_name, img_name = line.split("/")
            if class_name not in classes:
                continue
            img_path = os.path.join(data_path, "images", class_name, img_name + ".jpg")
            self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(data_path, classes, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = Food101Subset(data_path, classes, split="train", transform=train_transforms)
    val_dataset   = Food101Subset(data_path, classes, split="test",  transform=val_transforms)

    # shuffle makes sure that the data is randomly sampled during training, which helps in better generalization of the model.
    # num_workers=2 allows for parallel data loading, which can speed up the training process by utilizing multiple CPU cores.
    # pin_memory=True is used to speed up the transfer of data from CPU to GPU,
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader