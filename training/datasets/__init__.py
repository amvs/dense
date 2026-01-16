from .mnist import get_mnist_loaders
from .kaggle import get_kaggle_loaders
from .kthtips2b import get_kthtips2b_loaders
from .outex import get_outex_loaders
from .base import split_train_val
from dense.helpers.logger import LoggerManager

Loaders = ["mnist", "smohsensadeghi/curet-dataset", "roustoumabdelmoula/textures-dataset", "saurabhshahane/barkvn50",
"prasunroy/natural-images", "liewyousheng/minc2500", "kthtips2b", "outex10", "outex12"]

def get_loaders(dataset, *args, **kwargs):
    if dataset not in Loaders:
        raise ValueError("[training.dataset]: dataset loaders for {dataset} is not supported") 
    if dataset == "mnist":
        loaders = get_mnist_loaders(*args, **kwargs)
    elif dataset == "kthtips2b":
        loaders = get_kthtips2b_loaders(*args, **kwargs)
    elif dataset.startswith("outex"):
        problem = kwargs.get("fold", 0)
        all_problems = outex.get_available_problems(kwargs['root_dir'])
        if problem > len(all_problems)-1:
            raise ValueError(f"[training.dataset]: only {len(all_problems)} problems available, but fold id {problem} was given.")
        logger = LoggerManager.get_logger()
        logger.info(f"[training.dataset]: Using Outex fold id {problem}, maps to problem {all_problems[problem]}")
        loaders = get_outex_loaders(problem_id = all_problems[problem], *args, **kwargs)
    else:
        loaders = get_kaggle_loaders(dataset, *args, **kwargs)
    return loaders

__all__ = ["get_loaders", "split_train_val"]