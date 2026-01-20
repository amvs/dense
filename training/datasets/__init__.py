from .mnist import get_mnist_loaders
from .kaggle import get_kaggle_loaders
from .base import split_train_val

Loaders = ["mnist", "smohsensadeghi/curet-dataset", "roustoumabdelmoula/textures-dataset", "saurabhshahane/barkvn50",
"prasunroy/natural-images", "kthtips2b", "puneet6060/intel-image-classification", "outex", "jmexpert/describable-textures-dataset-dtd"]

def get_loaders(dataset, *args, **kwargs):
    if dataset not in Loaders:
        raise ValueError(f"[training.dataset]: dataset loaders for {dataset} is not supported") 
    if dataset == "mnist":
        loaders = get_mnist_loaders(*args, **kwargs)
    elif dataset == "kthtips2b":
        from .kthtips2b import get_kthtips2b_loaders
        loaders = get_kthtips2b_loaders(*args, **kwargs)
    elif dataset == "outex":
        from .outex import get_outex_loaders
        loaders = get_outex_loaders(*args, **kwargs)
    else:
        loaders = get_kaggle_loaders(dataset, *args, **kwargs)
    return loaders

__all__ = ["get_loaders", "split_train_val"]