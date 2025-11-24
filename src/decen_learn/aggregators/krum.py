# src/decen_learn/aggregators/krum.py
import warnings
import torch
from .base import BaseAggregator, AggregationResult
from .geomedian import GeoMedianAggregator

class KrumAggregator(BaseAggregator):
    """Krum aggregator"""
    
    def __init__(
        self,
        num_byzantine: int = 0,
        byzantine_fraction: float = 1/3,
        num_select: int = 1
    ):
        super().__init__(
            num_byzantine=num_byzantine,
            byzantine_fraction=byzantine_fraction,
        )
        self.num_select = num_select
    
    def aggregate(self, vectors: torch.Tensor) -> AggregationResult:
        m, d = vectors.shape
        f = self.num_byzantine if self.num_byzantine > 0 else int(self.byzantine_fraction * m)
        
        if not (0 <= f < m // 2):
            raise ValueError(f"Invalid f={f} for m={m} vectors")
        if not (m - f - 2 > 0):
            warnings.warn(
                f"f={f}, m={m} violates Krum condition m - f - 2 > 0. "
                "Fallback to Geo-Median.",
                RuntimeWarning,
            )
            geomed = GeoMedianAggregator()
            return geomed.aggregate(vectors)
    
        scores = self._compute_scores(vectors, f)
        selected_idx = int(torch.argmin(scores).item())
        
        return AggregationResult(
            vector=vectors[selected_idx].clone(),
            selected_index=selected_idx,
            metadata={"scores": scores.detach().cpu().numpy()}
        )
    
    def _compute_scores(self, vectors: torch.Tensor, f: int) -> torch.Tensor:
        m = vectors.shape[0]
        nb = m - f - 2
        
        vec = vectors
        if vec.dtype != torch.float64:
            vec = vec.to(torch.float64)
        sq_norms = torch.sum(vec ** 2, dim=1, keepdim=True)
        dists = sq_norms + sq_norms.T - 2.0 * (vec @ vec.T)
        dists = torch.clamp(dists, min=0.0)
        dists.fill_diagonal_(0.0)
        
        if nb <= 0:
            return dists.sum(dim=1).to(vectors.dtype)
    
        scores = torch.empty(m, device=vectors.device, dtype=torch.float64)
        for i in range(m):
            d_i = torch.cat([dists[i, :i], dists[i, i+1:]])
            topk = torch.topk(d_i, k=nb, largest=False).values
            scores[i] = topk.sum()
        
        return scores.to(vectors.dtype)
