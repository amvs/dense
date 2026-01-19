"""
Test script to verify the refactored WPH classifier structure works correctly.
"""
import torch
import numpy as np

print("Testing refactored WPH classifiers...")

# Test 1: Import all classifier types
print("\n1. Testing imports...")
from wph.classifiers import LinearClassifier, HyperNetworkClassifier, SVMClassifier, PCAClassifier
from wph.wph_model import WPHClassifier, WPHSvm, WPHPca
from wph.classifier_factory import (
    create_linear_wph_classifier,
    create_hypernetwork_wph_classifier,
    create_svm_wph_classifier,
    create_pca_wph_classifier
)
print("✓ All imports successful")

# Test 2: Create individual classifiers
print("\n2. Testing individual classifier creation...")
input_dim = 100
num_classes = 10

linear_clf = LinearClassifier(input_dim=input_dim, num_classes=num_classes, use_batch_norm=True)
print(f"✓ LinearClassifier created: {linear_clf}")

hypernet_clf = HyperNetworkClassifier(num_classes=num_classes, metadata_dim=32, hidden_dim=64)
print(f"✓ HyperNetworkClassifier created: {hypernet_clf}")

svm_clf = SVMClassifier(input_dim=input_dim, num_classes=num_classes)
print(f"✓ SVMClassifier created: {svm_clf}")

pca_clf = PCAClassifier(input_dim=input_dim, num_classes=num_classes, n_components=3)
print(f"✓ PCAClassifier created: {pca_clf}")

# Test 3: Test linear classifier forward pass
print("\n3. Testing LinearClassifier forward pass...")
batch_size = 5
features = torch.randn(batch_size, input_dim)
logits = linear_clf(features)
assert logits.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
print(f"✓ Linear forward pass successful: input {features.shape} -> output {logits.shape}")

# Test 4: Test hypernetwork classifier forward pass
print("\n4. Testing HyperNetworkClassifier forward pass...")
feature_metadata = torch.randn(input_dim, 32)
hypernet_clf.set_feature_metadata(feature_metadata)
logits = hypernet_clf(features)
assert logits.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
print(f"✓ HyperNetwork forward pass successful: input {features.shape} -> output {logits.shape}")

# Test 5: Test SVM classifier (passthrough)
print("\n5. Testing SVMClassifier...")
output = svm_clf(features)
assert torch.equal(output, features), "SVM classifier should be passthrough"
print(f"✓ SVM classifier passthrough successful")

# Test 6: Test PCA classifier
print("\n6. Testing PCAClassifier...")
# Fit the classifier first (need enough samples per class for PCA)
train_features = np.random.randn(100, input_dim)
train_labels = np.random.randint(0, num_classes, size=100)
pca_clf.fit(train_features, train_labels)
print("✓ PCA classifier fitted")

# Test forward pass
with torch.no_grad():
    log_probs = pca_clf(features)
    assert log_probs.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {log_probs.shape}"
print(f"✓ PCA forward pass successful: input {features.shape} -> output {log_probs.shape}")

# Test 7: Test factory functions
print("\n7. Testing factory functions...")
print("✓ Factory functions available:")
print(f"  - create_linear_wph_classifier")
print(f"  - create_hypernetwork_wph_classifier")
print(f"  - create_svm_wph_classifier")
print(f"  - create_pca_wph_classifier")

# Test 8: Verify backward compatibility classes exist
print("\n8. Testing backward compatibility...")
print(f"✓ WPHSvm class available: {WPHSvm}")
print(f"✓ WPHPca class available: {WPHPca}")

print("\n" + "="*50)
print("ALL TESTS PASSED! ✓")
print("="*50)
print("\nThe refactored WPH classifier structure is working correctly.")
print("See wph/classifiers/README.md for usage documentation.")
