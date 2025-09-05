from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch

class CuretDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = self.root_dir / cls
            for img_file in sorted(cls_folder.glob("*.png")) + sorted(cls_folder.glob("*.jpg")) + sorted(cls_folder.glob("*.bmp")):
                self.image_paths.append(img_file)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def get_curet_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = CuretDataset("./data/curetgrey", transform=transform)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 46, 46], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader