"""
Scale Augmentation for Scattering Transform Datasets.

Applies dilation/scaling to images with factors {1, sqrt(2), 2, 2*sqrt(2)}.
This creates 4 versions of each image, multiplying the dataset size by 4.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import math


class ScaleAugmentation:
    """
    Transform that applies scale augmentation by dilating images.
    
    Scaling factors: {1, sqrt(2), 2, 2*sqrt(2)}
    For each image, creates 4 versions with different scales.
    """
    def __init__(self, scale_factors=None):
        """
        Args:
            scale_factors: List of scale factors. Default: [1, sqrt(2), 2, 2*sqrt(2)]
        """
        if scale_factors is None:
            self.scale_factors = [1.0, math.sqrt(2), 2.0, 2 * math.sqrt(2)]
        else:
            self.scale_factors = scale_factors
    
    def __call__(self, img, target_size):
        """
        Apply scale augmentation to image.
        
        Args:
            img: PIL Image or torch.Tensor [C, H, W]
            target_size: Target size after scaling (for consistent output size)
            
        Returns:
            List of images with different scales (as tensors if input was tensor)
        """
        # Handle tensor input (after normalization, images are tensors)
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Convert tensor to PIL for scaling operations
            # Tensor is [C, H, W], need to convert to PIL
            img_pil = F.to_pil_image(img)
        else:
            img_pil = img
        
        augmented_images = []
        for scale in self.scale_factors:
            # Calculate new size: scale the image
            # This simulates dilation: larger scale = more zoomed in
            w, h = img_pil.size
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image with scale factor (dilation)
            scaled_img = img_pil.resize((new_w, new_h), Image.BILINEAR)
            
            # Center crop back to original size (or resize to target_size)
            # This simulates viewing the image at different scales
            if scale >= 1.0:
                # For scale >= 1, center crop to show zoomed-in view
                crop_size = min(new_w, new_h)
                left = (new_w - crop_size) // 2
                top = (new_h - crop_size) // 2
                scaled_img = scaled_img.crop((left, top, left + crop_size, top + crop_size))
            
            # Resize to target size for consistency
            scaled_img = scaled_img.resize((target_size, target_size), Image.BILINEAR)
            
            # Convert back to tensor if original was tensor
            if is_tensor:
                scaled_img = F.to_tensor(scaled_img)
            
            augmented_images.append(scaled_img)
        
        return augmented_images


class ScaleAugmentedDataset(Dataset):
    """
    Dataset wrapper that applies scale augmentation to each sample.
    Creates 4 versions of each image with different scales.
    """
    def __init__(self, base_dataset, scale_factors=None, target_size=200):
        """
        Args:
            base_dataset: Base dataset to augment
            scale_factors: List of scale factors. Default: [1, sqrt(2), 2, 2*sqrt(2)]
            target_size: Target size for resized images (default: 200)
        """
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.scale_aug = ScaleAugmentation(scale_factors)
        
        # Each sample in base dataset becomes 4 samples (one per scale factor)
        self.num_scales = len(self.scale_aug.scale_factors)
    
    def __len__(self):
        """Return augmented dataset size (4x original)."""
        return len(self.base_dataset) * self.num_scales
    
    def __getitem__(self, idx):
        """
        Get augmented sample.
        
        Args:
            idx: Index in augmented dataset
            
        Returns:
            (image, label): Augmented image (PIL Image) and label
        """
        # Map augmented index to base dataset index and scale factor
        base_idx = idx // self.num_scales
        scale_idx = idx % self.num_scales
        
        # Get original sample from base dataset
        # Temporarily disable transform to get raw PIL image for scale augmentation
        # Scale augmentation must happen BEFORE normalization
        original_transform = getattr(self.base_dataset, 'transform', None)
        self.base_dataset.transform = None
        
        # Get original sample as PIL Image (before any transforms)
        img_pil, label = self.base_dataset[base_idx]
        
        # Restore transform (will be applied after scale augmentation)
        self.base_dataset.transform = original_transform
        
        # Apply scale augmentation to PIL image
        # This creates 4 versions with different scales (all PIL Images)
        augmented_images = self.scale_aug(img_pil, self.target_size)
        
        # Get the specific scale version (PIL Image)
        scaled_img_pil = augmented_images[scale_idx]
        
        # Apply normalization transform if base dataset has one
        # This converts PIL -> Tensor -> Normalize
        if original_transform is not None:
            scaled_img = original_transform(scaled_img_pil)
        else:
            # No transform, return PIL image
            scaled_img = scaled_img_pil
        
        return scaled_img, label
    
    @property
    def classes(self):
        """Forward classes from base dataset."""
        return self.base_dataset.classes
    
    @property
    def class_to_idx(self):
        """Forward class_to_idx from base dataset."""
        return self.base_dataset.class_to_idx
