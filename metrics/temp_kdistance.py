import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKDistance:
    def __init__(self, method: str = 'euclidean',aggregate: str = 'mean'):
        self.method = method
        self.aggregate = aggregate
        self._methods = {
            'euclidean': self._euclidean,
            'cosine': self._cosine,
            'correlation': self._correlation,
            'chebyshev': self._chebyshev,
            'minkowski': self._minkowski,
            'cityblock': self._cityblock
        }
        self._aggregs = {
            'mean': self._mean,
            'median': self._median,
            'robust_mean': self._robust_mean
        }

    def __call__(self, arr1: torch.Tensor, arr2: torch.Tensor, k_range=[1, 5, 10, 100]) -> dict[str, float]:
        if self.method not in self._methods:
            raise ValueError(f"{self.method} not in list of defined methods. Please choose from {list(self._methods.keys())}")
        if self.aggregate not in self._aggregs:
            raise ValueError(f"{self.aggregate} not in list of defined aggregations. Please choose from {list(self._aggregs.keys())}")
        
        # check if arr1 and arr2 are 2D or 3D tensors and raise error if not
        if arr1.ndim not in [2, 3]:
            raise ValueError(f"arr1 should be a 2D or 3D tensor, but got {arr1.ndim}D tensor")
        if arr2.ndim not in [2, 3]:
            raise ValueError(f"arr2 should be a 2D or 3D tensor, but got {arr2.ndim}D tensor")


        if arr1.ndim == 3:
            arr1 = self._aggregs[self.aggregate](arr1)
        if arr2.ndim == 3:
            arr2 = self._aggregs[self.aggregate](arr2)
        matrix = self._methods[self.method](arr1, arr2)
        ranks = self._compute_diag_ranks(matrix)
        mean_ranks = torch.mean(ranks, dim=0)
        dict_score = {}
        for k in k_range:
            number_values = matrix.shape[0] * (k/100)
            r = (ranks <= number_values).sum()
            r = (r/matrix.shape[0]) * 100
            dict_score['top'+str(k)] = r
        dict_score['mean_ranks'] = (mean_ranks/matrix.shape[0]) * 100
        r_exact = (ranks == 0).sum()
        dict_score['exact_matching'] = (r_exact/matrix.shape[0]) * 100
        return dict_score

    @staticmethod
    def _compute_diag_ranks(matrix: torch.Tensor) -> torch.Tensor:
        _, indices = torch.sort(matrix, dim=1)
        ranks = torch.zeros(matrix.shape[0], dtype=torch.int64)
        for i in range(matrix.shape[0]):
            ranks[i] = (indices[i] == i).nonzero()[0].item() + 1  # to have index starting at 1
        return ranks

    @staticmethod
    def _euclidean(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
        return torch.norm(arr1 - arr2, p=2, dim=-1)

    @staticmethod
    def _cosine(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(arr1, arr2, dim=-1)

    @staticmethod
    def _correlation(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
        mean1 = torch.mean(arr1, dim=-1, keepdim=True)
        mean2 = torch.mean(arr2, dim=-1, keepdim=True)
        centered1 = arr1 - mean1
        centered2 = arr2 - mean2
        return 1 - torch.sum(centered1 * centered2, dim=-1) / torch.norm(centered1, p=2, dim=-1) / torch.norm(centered2, p=2, dim=-1)

    @staticmethod
    def _chebyshev(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(arr1 - arr2), dim=-1)[0]

    @staticmethod
    def _minkowski(arr1: torch.Tensor, arr2: torch.Tensor, p: float = 3) -> torch.Tensor:
        return torch.norm(arr1 - arr2, p=p, dim=-1)

    @staticmethod
    def _cityblock(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(arr1 - arr2), dim=-1)


    @staticmethod
    def _mean(emb: torch.Tensor) -> torch.Tensor:
        return torch.mean(emb, dim=0)
    @staticmethod
    def _median(emb: torch.Tensor) -> torch.Tensor:
        return torch.median(emb, dim=0)[0]
    @staticmethod
    def _robust_mean(emb: torch.Tensor) -> torch.Tensor:
        log_embeddings = torch.log(emb + 1e-8)
        avg_log_embeddings = torch.mean(log_embeddings, dim=0)
        return torch.exp(avg_log_embeddings)


