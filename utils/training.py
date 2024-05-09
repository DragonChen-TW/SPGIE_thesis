from torch import nn

def print_model_summary(model: nn.Module):
    """Override as needed"""
    print(
        'Model: \n%s\nParameters: %i' %
        (model, sum(p.numel() for p in model.parameters()))
    )