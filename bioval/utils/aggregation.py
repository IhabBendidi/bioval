import torch

def get_aggregation_functions() -> dict:
    methods = {
        'mean': mean,
        'median': median,
        'robust_mean': robust_mean
    }
    return methods
def mean(emb: torch.Tensor) -> torch.Tensor:
    if emb.ndim == 3:
        return torch.mean(emb, dim=1)
    else:        return emb
    
def median(emb: torch.Tensor) -> torch.Tensor:
    if emb.ndim == 3:
        return torch.median(emb, dim=1)[0]
    else:
        return emb
    
def robust_mean(emb: torch.Tensor) -> torch.Tensor:
    if emb.ndim == 3:
        log_embeddings = torch.log(emb + 1e-8)
        avg_log_embeddings = torch.mean(log_embeddings, dim=1)
        return torch.exp(avg_log_embeddings)
    else:
        return emb
