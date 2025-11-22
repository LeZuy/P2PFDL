# tests/test_aggregators.py
import numpy as np
import pytest
from decen_learn.aggregators import KrumAggregator, TverbergAggregator

class TestKrumAggregator:
    
    def test_basic_aggregation(self):
        """Krum should select the most central vector."""
        vectors = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],  # Outlier
        ])
        
        agg = KrumAggregator(byzantine_fraction=1/3)
        result = agg(vectors)
        
        # Should not select the outlier
        assert result.selected_index != 3
        assert np.linalg.norm(result.vector) < 1.0
    
    def test_all_identical(self):
        """Krum should handle identical vectors."""
        vectors = np.ones((5, 10))
        agg = KrumAggregator(byzantine_fraction=1/3)
        result = agg(vectors)
        
        np.testing.assert_array_equal(result.vector, np.ones(10))
    
    @pytest.mark.parametrize("n,f", [(10, 1/3), (20, 1/3), (50, 1/3)])
    def test_byzantine_tolerance(self, n, f):
        """Krum should tolerate up to f Byzantine vectors."""
        nf = int(n * f)
        rng = np.random.default_rng(42)
        
        # Honest vectors clustered around origin
        honest = rng.normal(0, 0.1, size=(n - nf, 2))
        # Byzantine vectors far away
        byzantine = rng.normal(100, 1, size=(nf, 2))
        
        vectors = np.vstack([honest, byzantine])
        rng.shuffle(vectors)
        
        agg = KrumAggregator(byzantine_fraction=f)
        result = agg(vectors)
        
        # Result should be close to origin
        assert np.linalg.norm(result.vector) < 1.0


class TestTverbergAggregator:
    
    def test_centerpoint_guarantee(self):
        """Centerpoint should have depth >= n/3."""
        rng = np.random.default_rng(42)
        vectors = rng.uniform(-1, 1, size=(30, 2))
        
        agg = TverbergAggregator()
        result = agg(vectors)
        
        # Verify depth guarantee
        assert result.metadata["depth_estimate"] >= len(vectors) // 3
    
    def test_collinear_points(self):
        """Should handle nearly collinear points."""
        vectors = np.array([
            [0.0, 0.0],
            [1.0, 0.001],
            [2.0, -0.001],
            [3.0, 0.002],
        ])
        
        agg = TverbergAggregator()
        result = agg(vectors)
        
        # Should return something reasonable
        assert not np.isnan(result.vector).any()