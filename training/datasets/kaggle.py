from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
from dense.helpers import LoggerManager
import kagglehub
import os

def get_kaggle_dataset(dataset: str) -> str:
    """
    Download (or reuse cached) Kaggle dataset via kagglehub.

    Parameters
    ----------
    dataset : str
        The Kaggle dataset identifier, e.g. "roustoumabdelmoula/textures-dataset".

    Returns
    -------
    str
        Local path to the dataset files.
    """
    os.environ['KAGGLEHUB_CACHE'] = './data'
    # kagglehub automatically caches datasets
    path = kagglehub.dataset_download(dataset)
    if not os.path.exists(path):
        raise RuntimeError(f"Dataset path {path} not found after download.")
    return path

class KaggleDataset(Dataset):
    def __init__(self, root_dir, deeper_path='', transform=None):
        self.root_dir = Path(root_dir) / deeper_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = self.root_dir / cls
            for img_file in sorted(cls_folder.glob("*.png")) + sorted(cls_folder.glob("*.JPG")) +  sorted(cls_folder.glob("*.jpg")) + sorted(cls_folder.glob("*.bmp")):
                self.image_paths.append(img_file)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # grayscale only
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def get_kaggle_loaders(dataset_name, resize, deeper_path, batch_size=64, train_ratio=0.8, worker_init_fn=None):
    logger = LoggerManager.get_logger()
    path = get_kaggle_dataset(dataset_name) # download if not exists
    logger.info(f"Load dataset {dataset_name} from Kaggle...")
    logger.info(f"Dataset path: {path} + {deeper_path}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize, resize)),      # resize shorter side
    ])
    #dataset = datasets.ImageFolder(root=path, transform=transform)
    dataset = KaggleDataset(path, deeper_path, transform=transform)
    mean, std = compute_mean_std(dataset)
    dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize, resize)),
        transforms.Normalize((mean,), (std,))  # only gray
    ])
    logger.info(f"mean: {mean}, std: {std}")

    # compute split sizes
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    # split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=True)
    nb_class = len(dataset.classes)
    sample_img, _ = train_dataset[0]
    logger.info(f"[Ratio:{train_ratio}] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    return train_loader, test_loader, nb_class, sample_img.shape

def compute_mean_std(dataset):
    """Compute mean and std for a grayscale dataset."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    sq_mean = 0.0
    num_pixels = 0

    for imgs, _ in loader:
        imgs = imgs.view(imgs.size(0), -1)  # flatten H*W
        mean += imgs.sum()
        sq_mean += (imgs ** 2).sum()
        num_pixels += imgs.numel()

    mean /= num_pixels
    std = (sq_mean / num_pixels - mean ** 2).sqrt()
    return mean.item(), std.item()


if __name__ == "__main__":
    # script to run and compute the statistics
    dataset_name = "smohsensadeghi/curet-dataset"  # replace with your path
    train_loader, test_loader, nb_class, img_shape = get_kaggle_loaders(dataset_name, resize=64, batch_size=64)
    #CURET dataset mean: 0.317774, std: 0.213433