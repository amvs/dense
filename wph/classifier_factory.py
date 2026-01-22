"""
Helper functions for creating WPHClassifier instances with different classifier types.

These convenience functions make it easier to create WPHClassifier instances
with the appropriate classifier module.
"""
from typing import Optional
from wph.classifiers import LinearClassifier, HyperNetworkClassifier, SVMClassifier, PCAClassifier
from wph.wph_model import WPHClassifier


def create_linear_wph_classifier(feature_extractor, num_classes: int, 
                                   use_batch_norm: bool = False, 
                                   copies: int = 1, noise_std: float = 0.01):
    """
    Create a WPHClassifier with a linear classifier.
    
    Args:
        feature_extractor: The WPH feature extractor (WPHModel or WPHModelDownsample).
        num_classes (int): Number of classes for classification.
        use_batch_norm (bool): Whether to use batch normalization.
        copies (int): Number of feature extractor copies to ensemble.
        noise_std (float): Standard deviation of noise added to filters for each copy.
        
    Returns:
        WPHClassifier: Configured classifier instance.
    """
    nb_moments = int(feature_extractor.nb_moments)
    classifier = LinearClassifier(
        input_dim=nb_moments,
        num_classes=num_classes,
        use_batch_norm=use_batch_norm
    )
    return WPHClassifier(
        feature_extractor=feature_extractor,
        classifier=classifier,
        copies=copies,
        noise_std=noise_std
    )


def create_hypernetwork_wph_classifier(feature_extractor, num_classes: int,
                                        metadata_dim: int, hidden_dim: int = 64,
                                        copies: int = 1, noise_std: float = 0.01):
    """
    Create a WPHClassifier with a hypernetwork classifier.
    
    Args:
        feature_extractor: The WPH feature extractor (WPHModel or WPHModelDownsample).
        num_classes (int): Number of classes for classification.
        metadata_dim (int): Dimensionality of the feature metadata.
        hidden_dim (int): Hidden dimension for the hypernetwork.
        copies (int): Number of feature extractor copies to ensemble.
        noise_std (float): Standard deviation of noise added to filters for each copy.
        
    Returns:
        WPHClassifier: Configured classifier instance.
    """
    classifier = HyperNetworkClassifier(
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        hidden_dim=hidden_dim
    )
    return WPHClassifier(
        feature_extractor=feature_extractor,
        classifier=classifier,
        copies=copies,
        noise_std=noise_std
    )


def create_svm_wph_classifier(feature_extractor, num_classes: Optional[int] = None,
                               copies: int = 1, noise_std: float = 0.01):
    """
    Create a WPHClassifier with an SVM classifier (placeholder for external SVM training).
    
    Args:
        feature_extractor: The WPH feature extractor (WPHModel or WPHModelDownsample).
        num_classes (int, optional): Number of classes (not used for SVM, kept for API consistency).
        copies (int): Number of feature extractor copies to ensemble.
        noise_std (float): Standard deviation of noise added to filters for each copy.
        
    Returns:
        WPHClassifier: Configured classifier instance.
    """
    nb_moments = int(feature_extractor.nb_moments)
    classifier = SVMClassifier(input_dim=nb_moments, num_classes=num_classes)
    return WPHClassifier(
        feature_extractor=feature_extractor,
        classifier=classifier,
        copies=copies,
        noise_std=noise_std
    )


def create_pca_wph_classifier(feature_extractor, num_classes: Optional[int] = None,
                               n_components: Optional[int] = None,
                               scale_features: bool = True, whiten: bool = False,
                               copies: int = 1, noise_std: float = 0.01):
    """
    Create a WPHClassifier with a PCA classifier.
    
    Args:
        feature_extractor: The WPH feature extractor (WPHModel or WPHModelDownsample).
        num_classes (int, optional): Number of classes (set after fitting).
        n_components (int, optional): Number of PCA components to keep.
        scale_features (bool): Whether to scale features before PCA.
        whiten (bool): Whether to whiten the PCA components.
        copies (int): Number of feature extractor copies to ensemble.
        noise_std (float): Standard deviation of noise added to filters for each copy.
        
    Returns:
        WPHClassifier: Configured classifier instance.
    """
    nb_moments = int(feature_extractor.nb_moments)
    classifier = PCAClassifier(
        input_dim=nb_moments,
        num_classes=num_classes,
        n_components=n_components,
        scale_features=scale_features,
        whiten=whiten
    )
    return WPHClassifier(
        feature_extractor=feature_extractor,
        classifier=classifier,
        copies=copies,
        noise_std=noise_std
    )
