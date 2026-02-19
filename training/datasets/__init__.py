from .mnist import get_mnist_loaders
from .kaggle import get_kaggle_loaders
from .kthtips2b import get_kthtips2b_loaders
from .outex import get_outex_loaders
from .tiny_imagenet import get_tinyimagenet_loaders
from .base import split_train_val
from dense.helpers.logger import LoggerManager

Loaders = ["mnist", "smohsensadeghi/curet-dataset", "roustoumabdelmoula/textures-dataset", "saurabhshahane/barkvn50",
"prasunroy/natural-images", "liewyousheng/minc2500", "kthtips2b", "outex10", "outex12", "jmexpert/describable-textures-dataset-dtd", "akash2sharma/tiny-imagenet"]

def get_loaders(dataset, *args, **kwargs):
    if dataset not in Loaders:
        raise ValueError("[training.dataset]: dataset loaders for {dataset} is not supported") 
    if dataset == "mnist":
        loaders = get_mnist_loaders(*args, **kwargs)
    elif dataset == "kthtips2b":
        loaders = get_kthtips2b_loaders(*args, **kwargs)
    elif dataset.startswith("outex"):
        loaders = get_outex_loaders(*args, **kwargs)
    elif dataset == "akash2sharma/tiny-imagenet":
        loaders = get_tinyimagenet_loaders(*args, **kwargs)
    else:
        loaders = get_kaggle_loaders(dataset, *args, **kwargs)
    return loaders

__all__ = ["get_loaders", "split_train_val"]