import torch

def get_aggregation_functions() -> dict:
    """Returns a dictionary containing aggregation functions.

    Returns:
        dict: a dictionary containing aggregation functions
    """
    methods = {
        'mean': mean,
        'median': median,
        'robust_mean': robust_mean
    }
    return methods

def mean(emb: torch.Tensor, control: bool = False) -> torch.Tensor:
    """Compute mean of embeddings.

    Args:
        emb (torch.Tensor): a tensor of shape (N, D) or (N, I, D)
        control (bool): whether to compute mean of shape (I, D) and output (D), 
            when input is of shape (I, D)

    Returns:
        torch.Tensor: a tensor of shape (N, D) or (D), depending on the value of control
    """
    if emb.ndim == 3:
        return torch.mean(emb, dim=1)
    elif emb.ndim == 2 and control:
        return torch.mean(emb, dim=0)
    else:        
        return emb
    
def median(emb: torch.Tensor, control: bool = False) -> torch.Tensor:
    """Compute median of embeddings.

    Args:
        emb (torch.Tensor): a tensor of shape (N, D) or (N, I, D)
        control (bool): whether to compute median of shape (I, D) and output (D), 
            when input is of shape (I, D)

    Returns:
        torch.Tensor: a tensor of shape (N, D) or (D), depending on the value of control
    """
    if emb.ndim == 3:
        return torch.median(emb, dim=1)[0]
    elif emb.ndim == 2 and control:
        return torch.median(emb, dim=0)[0]
    else:
        return emb
    
def robust_mean(emb: torch.Tensor, control: bool = False) -> torch.Tensor:
    """Compute robust mean of embeddings.

    Args:
        emb (torch.Tensor): a tensor of shape (N, D) or (N, I, D)
        control (bool): whether to compute robust mean of shape (I, D) and output (D), 
            when input is of shape (I, D)

    Returns:
        torch.Tensor: a tensor of shape (N, D) or (D), depending on the value of control
    """
    if emb.ndim == 3:
        log_embeddings = torch.log(emb + 1e-8)
        avg_log_embeddings = torch.mean(log_embeddings, dim=1)
        return torch.exp(avg_log_embeddings)
    elif emb.ndim == 2 and control:
        log_embeddings = torch.log(emb + 1e-8)
        avg_log_embeddings = torch.mean(log_embeddings, dim=0)
        return torch.exp(avg_log_embeddings)
    else:
        return emb

