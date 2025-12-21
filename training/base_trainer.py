import torch
from dense.helpers import LoggerManager
def regularization_loss(current_params, original_params, lambda_reg=1e-3):
    '''
    It is caller's responsibility to ensure that
    `current_params` and `original_params` have the same length and
    correspond to each other in order.
    '''
    reg_loss = 0.
    #num_params = 0

    for p, p0 in zip(current_params, original_params):
        reg_loss += (torch.abs(p - p0)**2).sum()
        #num_params += p.numel()

    #assert num_params > 0, "Model has no parameters for regularization; can't normalize by num params."
    reg_loss = lambda_reg * (reg_loss) # / num_params)

    return reg_loss

def train_one_epoch(model, loader, optimizer, base_loss, device, original_params=None, lambda_reg=None, vmap_chunk_size=None, r=None):
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
            reg_loss_value = regularization_loss(model.fine_tuned_params(), original_params, lambda_reg)
            loss += reg_loss_value
        loss.backward()
        optimizer.step()
        # if original_params is not None:
        #     project_weights(model, original_params, r)

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

# def project_weights(model, original_params, r):
#     logger = LoggerManager.get_logger()
#     with torch.no_grad():
#         dist_sq = 0.0

#         for p, p0 in zip(model.fine_tuned_params(), original_params):
#             if p.requires_grad:
#                 diff = p - p0
#                 dist_sq += torch.sum(torch.abs(diff) ** 2)

#         dist = torch.sqrt(dist_sq)

#         if dist > r:
#             logger.info(f"Projecting weights back to the radius {r:.4f} (current distance: {dist:.4f})")
#             scale = r / dist
#             for p, p0 in zip(model.fine_tuned_params(), original_params):
#                 if p.requires_grad:
#                     p.copy_(p0 + scale * (p - p0))