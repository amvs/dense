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

    Usage Example:
    # A function factory that get parametric model and return forward function
    # Note:  The returned function must take arbitary number of tensors as parameters
    #        And call to this forward function by 
    #           module(*list) or module(t1, t2, t3)

    def funcFactory(self, conv):
        def module(*inputs):
            imgs = torch.cat(inputs, dim=1)
            img_c = imgs.to(torch.complex64)
            result = self.nonLinear(conv(img_c))
            return result
        return module  
    
    # Suppose we build a module_list using funcFactory in init
    # so now, in the forward pass, we can do
    inputs = [img]
    for module in self.module_list:
        result = checkpoint(module, *inputs)
        # possibly collect the result

    """
    return CheckpointFunction.apply(func, *args)

