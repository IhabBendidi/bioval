import torch 
import torch.nn.functional as F
import ot 
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from piq import KID


# TODO : Fix error in sliced wasserstein distance

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

def get_distributed_distance_functions() -> dict:
    """
    Returns a dictionary of distance functions that can be used in distributed_distance.

    Returns:
        dict: A dictionary of distance functions.
    """
    methods = {'mmd': scalar_mmd, 
               'kid': compute_KID}
    return methods

def compute_KID(x,y,degree: int = 3):

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #x=torch.tensor(x).to(device)
    #y=torch.tensor(y).to(device)
    kid_metric=KID(degree)
    kid= kid_metric(x,y)
    kid=kid.item()
    return kid
"""
# Takes numpy array as input, degree polynomial is  3 by default can change it with degree=int, can also change gamma if you want
def KID(x, y, degree: int = 3):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    x_kernel = rbf_kernel(x.cpu(), gamma=1 / degree)
    y_kernel = rbf_kernel(y.cpu(), gamma=1 / degree)
    
    xy_kernel = rbf_kernel(x.cpu(), y.cpu(), gamma=1 / degree)

    mmd = (x_kernel.sum() / (x.size(0) ** 2)) + (y_kernel.sum() / (y.size(0) ** 2)) - 2 * (
        xy_kernel.sum() / (x.size(0) * y.size(0)))
    
    mmd = torch.tensor(mmd)
    return mmd.item()
"""

def sliced_wasserstein_distance(x, y, n_projections=50):
    device = x.device
    n_points = x.shape[0]

    # Generate random projection vectors
    random_projections = torch.randn(n_points, n_projections, device=device)

    # Project the data onto the random vectors
    x_proj = torch.matmul(x, random_projections)
    y_proj = torch.matmul(y, random_projections)

    # Sort the projected data along the random projection axis
    x_proj_sorted, _ = torch.sort(x_proj, dim=0)
    y_proj_sorted, _ = torch.sort(y_proj, dim=0)

    # Compute the Sliced Wasserstein Distance
    swd = torch.sqrt(torch.sum((x_proj_sorted - y_proj_sorted)**2)) / n_projections

    return swd.item()


def mmd_distance(x, y, gamma):
    xx = torch.exp(-gamma * torch.cdist(x, x))
    xy = torch.exp(-gamma * torch.cdist(x, y))
    yy = torch.exp(-gamma * torch.cdist(y, y))

    return xx.mean() + yy.mean() - 2 * xy.mean()

def scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = torch.tensor(float('nan'))
        return mmd

    return torch.mean(torch.stack(list(map(lambda x: safe_mmd(target, transport, x), gammas))), dim=0)

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
