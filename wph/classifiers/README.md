# WPH Classifiers Module

This module provides a modular, lightweight architecture for classification using WPH (Wavelet Phase Harmonics) features. Each classifier type is implemented in its own file for better organization and maintainability.

## Structure

```
wph/
├── classifiers/
│   ├── __init__.py                    # Exports all classifier types
│   ├── linear_classifier.py          # Standard linear classifier with optional batch norm
│   ├── hypernetwork_classifier.py    # HyperNetwork-based classifier
│   ├── svm_classifier.py              # SVM classifier placeholder
│   └── pca_classifier_wrapper.py     # PCA/affine subspace classifier
├── classifier_factory.py              # Convenience factory functions
└── wph_model.py                       # Core WPH models and lightweight WPHClassifier wrapper
```

## Usage

### Option 1: Using WPHClassifier directly

```python
from wph.wph_model import WPHModel, WPHClassifier
from wph.classifiers import LinearClassifier

# Create feature extractor
feature_extractor = WPHModel(filters=filters, J=3, L=4, ...)

# Create classifier
classifier = LinearClassifier(
    input_dim=int(feature_extractor.nb_moments),
    num_classes=10,
    use_batch_norm=True
)

# Combine them
model = WPHClassifier(
    feature_extractor=feature_extractor,
    classifier=classifier,
    copies=1,
    noise_std=0.01
)

# Use the model
logits = model(x)
```

### Option 2: Using factory functions

```python
from wph.wph_model import WPHModel
from wph.classifier_factory import create_linear_wph_classifier

# Create feature extractor
feature_extractor = WPHModel(filters=filters, J=3, L=4, ...)

# Create model with factory function
model = create_linear_wph_classifier(
    feature_extractor=feature_extractor,
    num_classes=10,
    use_batch_norm=True,
    copies=1
)

# Use the model
logits = model(x)
```

### Option 3: For SVM or PCA classifiers

For SVM and PCA classifiers, create the appropriate classifier wrapper and use WPHClassifier directly:

```python
from wph.wph_model import WPHModel, WPHClassifier
from wph.classifiers import PCAClassifierWrapper

feature_extractor = WPHModel(filters=filters, J=3, L=4, ...)

# Create PCA classifier wrapper
pca_classifier = PCAClassifierWrapper(
    input_dim=int(feature_extractor.nb_moments),
    n_components=50,
    scale_features=True
)

model = WPHClassifier(
    feature_extractor=feature_extractor,
    classifier=pca_classifier,
    copies=1
)

# Extract features and fit PCA
train_features = model.extract_features(train_data)
model.classifier.fit(train_features.cpu().numpy(), train_labels)

# Predict
predictions = model.classifier.predict(torch.tensor(test_features))
```

## Classifier Types

### LinearClassifier
Standard linear layer with optional batch normalization.

**Parameters:**
- `input_dim`: Dimension of input features
- `num_classes`: Number of output classes
- `use_batch_norm`: Whether to apply batch normalization before classification

**Use case:** Standard classification tasks

### HyperNetworkClassifier
Generates classification weights from feature metadata using a hypernetwork.

**Parameters:**
- `num_classes`: Number of output classes
- `metadata_dim`: Dimensionality of feature metadata
- `hidden_dim`: Hidden layer size in the hypernetwork

**Use case:** When you want to leverage structural information about features

**Note:** Requires calling `set_feature_metadata()` before forward pass.

### SVMClassifier
Placeholder for external SVM training using sklearn.

**Parameters:**
- `input_dim`: Dimension of input features
- `num_classes`: Number of output classes (optional, for consistency)

**Use case:** When using external sklearn SVM for classification

**Note:** Use `extract_features()` to get features for sklearn SVM training.

### PCAClassifierWrapper
Wraps the PCA/affine subspace classifier for PyTorch compatibility.

**Parameters:**
- `input_dim`: Dimension of input features
- `num_classes`: Number of output classes (optional, set after fitting)
- `n_components`: Number of PCA components to keep per class
- `scale_features`: Whether to standardize features
- `whiten`: Whether to whiten PCA components

**Use case:** Classification via distance to learned affine subspaces

**Note:** Must call `fit()` before using for prediction.

## WPHClassifier Features

The unified `WPHClassifier` class provides:

1. **Ensemble support**: Multiple copies of feature extractor with noise-augmented filters
2. **Trainable ensemble weights**: Learned averaging of multiple feature extractors
3. **Flexible training**: Freeze/unfreeze feature extractor, classifier, or spatial attention independently
4. **Feature extraction**: Access to intermediate features for analysis via `extract_features()` method
5. **Pluggable classifiers**: Works with any classifier module (LinearClassifier, PCAClassifierWrapper, SVMClassifier, etc.)

### Key Methods

- `forward(x, vmap_chunk_size=None, return_feats=False)`: Standard forward pass
- `extract_features(x, vmap_chunk_size=None)`: Extract features without classification
- `set_trainable(parts)`: Control which parts are trainable
  - `parts = {'feature_extractor': True/False, 'classifier': True/False, 'spatial_attn': True/False}`
- `fine_tuned_params()`: Get list of trainable parameters from feature extractor

### Properties

- `nb_moments`: Number of features produced by the feature extractor

## Migration Guide

The refactoring consolidated all classification into a single `WPHClassifier` class.
Previously separate `WPHSvm` and `WPHPca` wrapper classes have been removed.

### Old approach (before simplification)
```python
# For PCA
from wph.wph_model import WPHPca
model = WPHPca(feature_extractor, copies=1, n_components=50)
model.set_classifier(pca_classifier)

# For SVM  
from wph.wph_model import WPHSvm
model = WPHSvm(feature_extractor, copies=1)
```

### New approach (after simplification)
```python
# For PCA
from wph.wph_model import WPHClassifier
from wph.classifiers import PCAClassifierWrapper

pca_classifier = PCAClassifierWrapper(
    input_dim=int(feature_extractor.nb_moments),
    n_components=50
)
model = WPHClassifier(
    feature_extractor=feature_extractor,
    classifier=pca_classifier,
    copies=1
)

# For SVM
from wph.classifiers import SVMClassifier

svm_classifier = SVMClassifier(input_dim=int(feature_extractor.nb_moments))
model = WPHClassifier(
    feature_extractor=feature_extractor,
    classifier=svm_classifier,
    copies=1
)
```

All functionality is preserved through `WPHClassifier` and the underlying classifier modules.

## Adding New Classifier Types

To add a new classifier type:

1. Create a new file in `wph/classifiers/` (e.g., `my_classifier.py`)
2. Implement your classifier as a `nn.Module` with:
   - `__init__(input_dim, num_classes, ...)` constructor
   - `forward(features)` method that takes features and returns logits
3. Add your classifier to `wph/classifiers/__init__.py`
4. (Optional) Add a factory function to `classifier_factory.py`

Example:
```python
# wph/classifiers/my_classifier.py
import torch
from torch import nn

class MyClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, my_param: float = 0.5):
        super().__init__()
        self.my_layer = nn.Linear(input_dim, num_classes)
        self.my_param = my_param
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.my_layer(features) * self.my_param
```

Then use it:
```python
from wph.classifiers import MyClassifier

classifier = MyClassifier(input_dim=1000, num_classes=10, my_param=0.7)
model = WPHClassifier(feature_extractor=fe, classifier=classifier)
```
