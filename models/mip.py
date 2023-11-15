import torch 

def expected_sin(x, x_var, compute_var=False):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    if compute_var:
        y_var = torch.clamp(0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2, min=0)
        return y, y_var
    else:
        return y