import torch
import time

def regularization_loss(current_params, original_params, lambda_reg=1e-3, normalize_reg=True):
    '''
    It is caller's responsibility to ensure that
    `current_params` and `original_params` have the same length and
    correspond to each other in order.
    '''
    reg_loss = 0.
    num_params = 0

    for p, p0 in zip(current_params, original_params):
        reg_loss += (torch.abs(p - p0)**2).sum()
        num_params += p.numel()
    if normalize_reg:
        assert num_params > 0, "Model has no parameters for regularization; can't normalize by num params."
        reg_loss = lambda_reg * (reg_loss / num_params)
    else:
        reg_loss = lambda_reg * reg_loss

    return reg_loss

def train_one_epoch(model, loader, optimizer, base_loss, device, original_params=None, lambda_reg=None, vmap_chunk_size=None, normalize_reg=True):
    model.train()
    total_loss = 0
    total_base_loss = 0
    total_reg_loss = 0
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if vmap_chunk_size is None:
            outputs = model(inputs)
        else:
            outputs = model.forward(inputs, vmap_chunk_size=vmap_chunk_size)
        base_loss_value = base_loss(outputs, targets) # Average loss in a batch
        loss = base_loss_value
        reg_loss_value = torch.tensor(0.0)
        if original_params is not None:
            if not hasattr(model, "fine_tuned_params") or not callable(getattr(model, "fine_tuned_params")):
                raise NotImplementedError(
                    "Model passed to train_one_epoch must implement a callable fine_tuned_params() method for regularization."
                )
            reg_loss_value = regularization_loss(model.fine_tuned_params(), original_params, lambda_reg, normalize_reg=normalize_reg)
            loss += reg_loss_value
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            
        optimizer.step()

        total_base_loss += base_loss_value.item() * inputs.size(0)
        total_reg_loss += reg_loss_value.item() * inputs.size(0)
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    avg_base_loss = total_base_loss / len(loader.dataset)
    avg_reg_loss = total_reg_loss / len(loader.dataset)
    avg_total_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return {
        "base_loss": avg_base_loss,
        "reg_loss": avg_reg_loss,
        "total_loss": avg_total_loss,
        "accuracy": accuracy
    }

def evaluate(model, loader, base_loss, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = base_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def recompute_bn_running_stats(model, loader, device, max_batches=100, logger=None, momentum=None):
    """Run forward-only passes to update BatchNorm running statistics.

    Performs up to `max_batches` forward passes with `torch.no_grad()` while the
    model is in train mode so BatchNorm modules update their running_mean/var
    without modifying parameters. This is robust to loaders with fewer than
    `max_batches` batches.

    If `momentum` is provided, temporarily set each BatchNorm module's
    `momentum` attribute to that value for the duration of the warmup and
    restore original values afterwards. Higher momentum (closer to 1.0)
    makes running stats track batch statistics more aggressively.
    """
    if logger:
        logger.log(f"Starting BN warmup for up to {max_batches} batches", data=True)
    # Optionally override BN momentum temporarily
    bn_modules = []
    old_momentums = []
    for m in model.modules():
        # cover BatchNorm variants
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            bn_modules.append(m)
            old_momentums.append(getattr(m, 'momentum', None))
            if momentum is not None:
                try:
                    m.momentum = momentum
                except Exception:
                    pass

    model.train()
    batches = 0
    start_t = time.time()
    try:
        with torch.no_grad():
            for batch in loader:
            # DataLoader typically yields (inputs, targets) tuples
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                elif isinstance(batch, dict):
                    inputs = batch.get("image") or batch.get("inputs") or next(iter(batch.values()))
                else:
                    inputs = batch
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                try:
                    inputs = inputs.to(device)
                except Exception:
                    pass
                try:
                    _ = model(inputs)
                except Exception:
                    # Best-effort: ignore forward errors during warmup
                    pass
                batches += 1
                if batches >= max_batches:
                    break
    finally:
        # Restore original BN momentums
        if len(bn_modules) > 0 and len(old_momentums) == len(bn_modules):
            for m, old in zip(bn_modules, old_momentums):
                try:
                    if old is None:
                        # if there was no attribute before, delete if possible
                        delattr(m, 'momentum')
                    else:
                        m.momentum = old
                except Exception:
                    pass
    if logger:
        logger.log(f"Completed BN warmup ({batches} batches) in {time.time()-start_t:.2f}s", data=True)