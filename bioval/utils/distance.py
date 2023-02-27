import torch 
import torch.nn.functional as F

def get_distance_functions() -> dict:
    """
    Returns a dictionary of distance functions that can be used in kNN search.

    Returns:
        dict: A dictionary of distance functions.
    """
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
    """
    Computes the Euclidean distance between two sets of vectors.

    Args:
        arr1 (torch.Tensor): A tensor of shape (n_samples_1, n_features).
        arr2 (torch.Tensor): A tensor of shape (n_samples_2, n_features).

    Returns:
        torch.Tensor: A tensor of shape (n_samples_1, n_samples_2) with the Euclidean distance between all pairs of vectors.
    """
    return torch.norm(arr1[:, None, :] - arr2[None, :, :], p=2, dim=-1)

def cosine(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two sets of vectors.

    Args:
        arr1 (torch.Tensor): A tensor of shape (n_samples_1, n_features).
        arr2 (torch.Tensor): A tensor of shape (n_samples_2, n_features).

    Returns:
        torch.Tensor: A tensor of shape (n_samples_1, n_samples_2) with the cosine similarity between all pairs of vectors.
    """
    return 1 - F.cosine_similarity(arr1[:, None, :], arr2[None, :, :], dim=-1)

def correlation(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Computes the correlation between two sets of vectors.

    Args:
        arr1 (torch.Tensor): A tensor of shape (n_samples_1, n_features).
        arr2 (torch.Tensor): A tensor of shape (n_samples_2, n_features).

    Returns:
        torch.Tensor: A tensor of shape (n_samples_1, n_samples_2) with the correlation between all pairs of vectors.
    """
    mean1 = torch.mean(arr1, dim=-1, keepdim=True)
    mean2 = torch.mean(arr2, dim=-1, keepdim=True)
    centered1 = arr1 - mean1
    centered2 = arr2 - mean2
    return 1 - torch.sum(centered1[:, None, :] * centered2[None, :, :], dim=-1) / torch.norm(centered1, p=2, dim=-1) / torch.norm(centered2, p=2, dim=-1)

def chebyshev(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Chebyshev distance between two sets of vectors.

    Args:
        arr1 (torch.Tensor): A tensor of shape (n_samples_1, n_features).
        arr2 (torch.Tensor): A tensor of shape (n_samples_2, n_features).

    Returns:
        torch.Tensor: A tensor of shape (n_samples_1, n_samples_2) with the Chebyshev distance between all pairs of vectors.
    """
    return torch.max(torch.abs(arr1[:, None, :] - arr2[None, :, :]), dim=-1)[0]
    

def minkowski(arr1: torch.Tensor, arr2: torch.Tensor, p: float = 3) -> torch.Tensor:
    """
    Calculates the Minkowski distance between two tensors.

    Args:
        arr1 (torch.Tensor): A tensor of shape (N1, C)
        arr2 (torch.Tensor): A tensor of shape (N2, C)
        p (float, optional): The order of the Minkowski distance. Default is 3.

    Returns:
        torch.Tensor: A tensor of shape (N1, N2) representing the Minkowski distance
        between each pair of elements in the input tensors.

    """
    return torch.norm(arr1[:, None, :] - arr2[None, :, :], p=p, dim=-1)


def cityblock(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the city block distance between two tensors.

    Args:
        arr1 (torch.Tensor): A tensor of shape (N1, C)
        arr2 (torch.Tensor): A tensor of shape (N2, C)

    Returns:
        torch.Tensor: A tensor of shape (N1, N2) representing the city block distance
        between each pair of elements in the input tensors.

    """
    return torch.sum(torch.abs(arr1[:, None, :] - arr2[None, :, :]), dim=-1)
