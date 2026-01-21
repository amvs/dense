"""
Balanced Batch Sampler for ensuring equal examples per class in each batch.
"""
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch contains equal number of examples from each class.
    
    Requirements:
    - batch_size must be a multiple of num_classes
    - Each batch will have batch_size // num_classes examples from each class
    
    Example:
        num_classes = 10, batch_size = 30
        Each batch: 3 examples from each class (30 // 10 = 3)
    """
    def __init__(self, dataset, batch_size, num_classes, shuffle=True, seed=None):
        """
        Args:
            dataset: Dataset with __getitem__ returning (image, label)
            batch_size: Must be multiple of num_classes
            num_classes: Number of classes
            shuffle: Whether to shuffle batches
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.seed = seed
        
        # Validate batch_size
        if batch_size % num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of num_classes ({num_classes}). "
                f"Each batch needs batch_size // num_classes = {batch_size // num_classes} examples per class."
            )
        
        self.examples_per_class = batch_size // num_classes
        
        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.class_to_indices[label].append(idx)
        
        # Check that we have enough examples per class
        min_examples = min(len(indices) for indices in self.class_to_indices.values())
        if min_examples < self.examples_per_class:
            raise ValueError(
                f"Not enough examples per class. Need at least {self.examples_per_class} per class, "
                f"but minimum is {min_examples}."
            )
        
        # Create batches: each batch has examples_per_class from each class
        self.batches = self._create_batches()
    
    def _create_batches(self):
        """Create batches ensuring equal examples per class."""
        batches = []
        
        # Create a copy of class indices for each class
        class_indices_copy = {
            cls: indices.copy() 
            for cls, indices in self.class_to_indices.items()
        }
        
        # Shuffle each class's indices
        rng = random.Random(self.seed) if self.seed is not None else random
        for cls in class_indices_copy:
            rng.shuffle(class_indices_copy[cls])
        
        # Create batches until we run out of examples
        while True:
            batch = []
            can_create_batch = True
            
            # Try to get examples_per_class from each class
            for cls in range(self.num_classes):
                if len(class_indices_copy[cls]) < self.examples_per_class:
                    can_create_batch = False
                    break
                # Take examples_per_class from this class
                batch.extend(class_indices_copy[cls][:self.examples_per_class])
                # Remove used indices
                class_indices_copy[cls] = class_indices_copy[cls][self.examples_per_class:]
            
            if not can_create_batch:
                break
            
            batches.append(batch)
        
        # Shuffle batches if requested
        if self.shuffle:
            rng = random.Random(self.seed) if self.seed is not None else random
            rng.shuffle(batches)
        
        return batches
    
    def __iter__(self):
        """Return iterator over batches."""
        return iter(self.batches)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.batches)


class BalancedSubsetSampler(Sampler):
    """
    Sampler for a subset that ensures balanced batches.
    Works with Subset datasets by mapping subset indices to original indices.
    """
    def __init__(self, subset, batch_size, num_classes, shuffle=True, seed=None):
        """
        Args:
            subset: torch.utils.data.Subset
            batch_size: Must be multiple of num_classes
            num_classes: Number of classes
            shuffle: Whether to shuffle batches
            seed: Random seed for reproducibility
        """
        self.subset = subset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.seed = seed
        
        # Validate batch_size
        if batch_size % num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of num_classes ({num_classes})"
            )
        
        self.examples_per_class = batch_size // num_classes
        
        # Group subset indices by class
        self.class_to_indices = defaultdict(list)
        for subset_idx in range(len(subset)):
            original_idx = subset.indices[subset_idx]
            _, label = subset.dataset[original_idx]
            self.class_to_indices[label].append(subset_idx)  # Store subset index
        
        # Check that we have enough examples per class
        min_examples = min(len(indices) for indices in self.class_to_indices.values())
        if min_examples < self.examples_per_class:
            raise ValueError(
                f"Not enough examples per class in subset. Need at least {self.examples_per_class} per class, "
                f"but minimum is {min_examples}."
            )
        
        # Create batches
        self.batches = self._create_batches()
    
    def _create_batches(self):
        """Create batches ensuring equal examples per class."""
        batches = []
        
        # Create a copy of class indices for each class
        class_indices_copy = {
            cls: indices.copy() 
            for cls, indices in self.class_to_indices.items()
        }
        
        # Shuffle each class's indices
        rng = random.Random(self.seed) if self.seed is not None else random
        for cls in class_indices_copy:
            rng.shuffle(class_indices_copy[cls])
        
        # Create batches until we run out of examples
        while True:
            batch = []
            can_create_batch = True
            
            # Try to get examples_per_class from each class
            for cls in range(self.num_classes):
                if len(class_indices_copy[cls]) < self.examples_per_class:
                    can_create_batch = False
                    break
                # Take examples_per_class from this class
                batch.extend(class_indices_copy[cls][:self.examples_per_class])
                # Remove used indices
                class_indices_copy[cls] = class_indices_copy[cls][self.examples_per_class:]
            
            if not can_create_batch:
                break
            
            batches.append(batch)
        
        # Shuffle batches if requested
        if self.shuffle:
            rng = random.Random(self.seed) if self.seed is not None else random
            rng.shuffle(batches)
        
        return batches
    
    def __iter__(self):
        """Return iterator over batches."""
        return iter(self.batches)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.batches)
