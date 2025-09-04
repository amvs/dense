import torch
from torch import nn
from torch.autograd import Function
class CheckpointFunction(Function):
    @staticmethod
    def forward(ctx, func, *args):
        ctx.func = func
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = func(*args)
        return outputs

    @staticmethod
    def backward(ctx, *upstream_grads):
        func = ctx.func
        inputs = ctx.saved_tensors
        detached_inputs = [inp.detach().requires_grad_(inp.requires_grad) for inp in inputs]
        with torch.enable_grad():
            outputs = func(*detached_inputs)
        torch.autograd.backward(outputs, upstream_grads)
        grads = [inp.grad for inp in detached_inputs]
        return (None, *grads)
    
def checkpoint(func, *args):
    """
    Checkpoint a func to avoid saving intermediate results. 
    This will save memory but increase running time.
    It assumes that args are all tensors. And it will
    save the inputs in the memory.

    Args:
        func: The sequence of operations to checkpoint.
        *args: The inputs to the func.
    
    Returns:
        The outputs of the func.

    Example:
    def MyModule(conv):
        def operation(img):
            result = self.non_linear(conv(img.to(torch.complex64)))
            img = torch.cat([img, result], dim=1)
            return img
        return operation
    
    # Suppose we build a module_list using MyModule in init
    # so now, in the forward pass, we can do
    for module in self.module_list:
        checkpoint(module, img)

    """
    return CheckpointFunction.apply(func, *args)

