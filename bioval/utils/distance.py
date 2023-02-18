import torch 

# TODO : Make this adaptable to choice of aggregated or distributional representation
def get_distance_functions() -> dict:
    methods = {
            'euclidean': euclidean,
            'cosine': cosine,
            'correlation': correlation,
            'chebyshev': chebyshev,
            'minkowski': minkowski,
            'cityblock': cityblock
        }
    return methods

def euclidean(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    return torch.norm(arr1[:, None, :] - arr2[None, :, :], p=2, dim=-1)

def cosine(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    return 1 - F.cosine_similarity(arr1[:, None, :], arr2[None, :, :], dim=-1)

def correlation(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    mean1 = torch.mean(arr1, dim=-1, keepdim=True)
    mean2 = torch.mean(arr2, dim=-1, keepdim=True)
    centered1 = arr1 - mean1
    centered2 = arr2 - mean2
    return 1 - torch.sum(centered1[:, None, :] * centered2[None, :, :], dim=-1) / torch.norm(centered1, p=2, dim=-1) / torch.norm(centered2, p=2, dim=-1)

def chebyshev(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(arr1[:, None, :] - arr2[None, :, :]), dim=-1)[0]

    
def minkowski(arr1: torch.Tensor, arr2: torch.Tensor, p: float = 3) -> torch.Tensor:
    return torch.norm(arr1[:, None, :] - arr2[None, :, :], p=p, dim=-1)

def cityblock(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(arr1[:, None, :] - arr2[None, :, :]), dim=-1)