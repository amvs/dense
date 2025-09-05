from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from dense.helpers import LoggerManager
def get_mnist_loaders(batch_size=64, train_ratio=0.8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    stdtrain_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    stdtest_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    if train_ratio == 1.0: # std setting
        train_loader = DataLoader(stdtrain_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(stdtest_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Merge into one dataset
        full_dataset = ConcatDataset([stdtrain_dataset, stdtest_dataset])

        # Extract all labels (ConcatDataset doesn't store `.targets` directly)
        targets = np.concatenate([np.array(stdtrain_dataset.targets), np.array(stdtest_dataset.targets)])

        # Stratified split
        indices = np.arange(len(targets))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=1-train_ratio,
            stratify=targets,
            random_state=42
        )

        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    nb_class = len(stdtrain_dataset.classes)
    sample_img, _ = train_dataset[0]
    logger = LoggerManager.get_logger()
    logger.info(f"[Ratio:{train_ratio}] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, # class: {nb_class}, shape {sample_img.shape}")
    return train_loader, test_loader, nb_class, sample_img.shape
