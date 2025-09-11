from .mnist import get_mnist_loaders
from .curet import get_curet_loaders
Loaders = {
    "mnist": get_mnist_loaders,
    "curet": get_curet_loaders
}
def get_loaders(dataset, *args, **kwargs):
    if dataset not in Loaders:
        raise ValueError("[training.dataset]: dataset loaders for {dataset} is not supported") 
    return Loaders[dataset](*args, **kwargs)

__all__ = ["get_loaders"]