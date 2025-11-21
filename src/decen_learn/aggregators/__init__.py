# src/decen_learn/aggregators/__init__.py
from typing import Dict, Type
from .base import BaseAggregator, AggregationResult
from .krum import KrumAggregator
from .tverberg import TverbergAggregator
from .mean import MeanAggregator
from .trimmed_mean import TrimmedMeanAggregator

AGGREGATORS: Dict[str, Type[BaseAggregator]] = {
    "mean": MeanAggregator,
    "krum": KrumAggregator,
    "tverberg": TverbergAggregator,
    "trimmed_mean": TrimmedMeanAggregator,
}

def get_aggregator(name: str, **kwargs) -> BaseAggregator:
    """Factory function for creating aggregators."""
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(AGGREGATORS.keys())}")
    return AGGREGATORS[name](**kwargs)