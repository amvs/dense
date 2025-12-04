import torch

def regularization_loss(current_params, original_params, lambda_reg=1e-3):
    '''
    It is caller's responsibility to ensure that
    `current_params` and `original_params` have the same length and
    correspond to each other in order.
    '''
    reg_loss = 0.
    for p, p0 in zip(current_params, original_params):
        reg_loss += (torch.abs(p - p0)**2).sum()
    reg_loss = lambda_reg * reg_loss
    return reg_loss

def train_one_epoch(model, loader, optimizer, base_loss, device, original_params=None, lambda_reg=None):
    model.train()
    total_loss = 0
    total_reg_loss = 0
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = base_loss(outputs, targets) # Average loss in a batch
        if original_params is not None:
            reg_loss = regularization_loss(model.fine_tuned_params(), original_params, lambda_reg)
            loss += reg_loss
            total_reg_loss += reg_loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0) # Total loss in a batch
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    avg_data_loss = (total_loss - total_reg_loss) / len(loader.dataset) # Average data loss, no reg loss
    accuracy = correct / len(loader.dataset) # Average loss
    return avg_data_loss, accuracy

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