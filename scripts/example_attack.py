#!/usr/bin/env python3
"""Example usage of migrated attack modules.

This script demonstrates how to use the refactored Byzantine attack
implementations in the decen_learn framework.
"""

import numpy as np
import torch
from pathlib import Path

# Mock imports (replace with actual paths in production)
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from decen_learn.attacks import MinMaxAttack, IPMAttack, LIEAttack


def create_mock_weights(num_layers: int = 3, dim_range: tuple = (50, 200)):
    """Create mock model weights for testing."""
    weights = {}
    for i in range(num_layers):
        dim = np.random.randint(*dim_range)
        weights[f"layer_{i}"] = np.random.randn(dim) * 0.1
    return weights


def example_basic_usage():
    """Basic usage of attack modules."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Attack Usage")
    print("=" * 80)
    
    # Simulate 5 honest neighbors
    honest_weights = [create_mock_weights() for _ in range(5)]
    attacker_weights = create_mock_weights()
    
    print(f"\nHonest neighbors: {len(honest_weights)}")
    print(f"Weight structure: {list(honest_weights[0].keys())}")
    
    # Test MinMax Attack
    print("\n--- MinMax Attack ---")
    minmax = MinMaxAttack(
        boosting_factor=1.0,
        gamma_init=20.0,
        oracle_type="minmax",
        perturb_kind="auto"
    )
    malicious_minmax = minmax.craft(honest_weights, attacker_weights)
    print(f"✓ Crafted malicious weights with {len(malicious_minmax)} layers")
    
    # Compute attack magnitude
    honest_mean = {
        k: np.mean([w[k] for w in honest_weights], axis=0)
        for k in honest_weights[0].keys()
    }
    magnitude = np.linalg.norm(
        np.concatenate([malicious_minmax[k] - honest_mean[k] for k in honest_mean.keys()])
    )
    print(f"  Attack magnitude: {magnitude:.4f}")
    
    # Test IPM Attack
    print("\n--- IPM Attack ---")
    ipm = IPMAttack(eps=0.5)
    malicious_ipm = ipm.craft(honest_weights, attacker_weights)
    print(f"✓ Crafted malicious weights with {len(malicious_ipm)} layers")
    
    magnitude = np.linalg.norm(
        np.concatenate([malicious_ipm[k] - honest_mean[k] for k in honest_mean.keys()])
    )
    print(f"  Attack magnitude: {magnitude:.4f}")
    
    # Test LIE Attack
    print("\n--- LIE Attack ---")
    lie = LIEAttack(num_byzantine=2, num_total=7)
    malicious_lie = lie.craft(honest_weights, attacker_weights)
    print(f"✓ Crafted malicious weights with {len(malicious_lie)} layers")
    
    magnitude = np.linalg.norm(
        np.concatenate([malicious_lie[k] - honest_mean[k] for k in honest_mean.keys()])
    )
    print(f"  Attack magnitude: {magnitude:.4f}")


def example_parameter_tuning():
    """Demonstrate parameter tuning for different scenarios."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Parameter Tuning")
    print("=" * 80)
    
    honest_weights = [create_mock_weights() for _ in range(10)]
    attacker_weights = create_mock_weights()
    
    # MinMax with different oracle types
    print("\n--- MinMax: Oracle Comparison ---")
    for oracle_type in ["minmax", "minsum"]:
        attack = MinMaxAttack(
            oracle_type=oracle_type,
            gamma_init=20.0,
            max_iter=100
        )
        malicious = attack.craft(honest_weights, attacker_weights)
        magnitude = np.linalg.norm(
            np.concatenate([v.flatten() for v in malicious.values()])
        )
        print(f"  {oracle_type:8s}: magnitude = {magnitude:.4f}")
    
    # MinMax with different perturbation directions
    print("\n--- MinMax: Perturbation Direction Comparison ---")
    for perturb_kind in ["unit", "std", "sign"]:
        attack = MinMaxAttack(
            perturb_kind=perturb_kind,
            gamma_init=20.0,
            max_iter=100
        )
        malicious = attack.craft(honest_weights, attacker_weights)
        magnitude = np.linalg.norm(
            np.concatenate([v.flatten() for v in malicious.values()])
        )
        print(f"  {perturb_kind:6s}: magnitude = {magnitude:.4f}")
    
    # LIE with different z values
    print("\n--- LIE: Z-value Comparison ---")
    for z_max in [0.5, 1.0, 1.5, 2.0]:
        attack = LIEAttack(z_max=z_max)
        malicious = attack.craft(honest_weights, attacker_weights)
        magnitude = np.linalg.norm(
            np.concatenate([v.flatten() for v in malicious.values()])
        )
        print(f"  z={z_max:.1f}: magnitude = {magnitude:.4f}")


def example_adversarial_scenario():
    """Simulate adversarial scenario with Byzantine fraction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Adversarial Scenario Simulation")
    print("=" * 80)
    
    # Network parameters
    total_nodes = 64
    byzantine_fraction = 0.33
    num_byzantine = int(total_nodes * byzantine_fraction)
    degree = 6  # Average connections per node
    
    print(f"\nNetwork Configuration:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Byzantine nodes: {num_byzantine} ({byzantine_fraction*100:.0f}%)")
    print(f"  Average degree: {degree}")
    
    # Simulate neighborhood (random honest neighbors)
    num_honest_neighbors = np.random.randint(3, degree)
    print(f"  Honest neighbors for Byzantine node: {num_honest_neighbors}")
    
    honest_weights = [create_mock_weights() for _ in range(num_honest_neighbors)]
    attacker_weights = create_mock_weights()
    
    # Test different attacks
    attacks = {
        "MinMax (minmax)": MinMaxAttack(oracle_type="minmax"),
        "MinMax (minsum)": MinMaxAttack(oracle_type="minsum"),
        "IPM (eps=0.5)": IPMAttack(eps=0.5),
        "IPM (eps=1.0)": IPMAttack(eps=1.0),
        "LIE": LIEAttack(num_byzantine=num_byzantine, num_total=total_nodes),
    }
    
    print("\n--- Attack Performance ---")
    for name, attack in attacks.items():
        malicious = attack.craft(honest_weights, attacker_weights)
        
        # Compute distance to honest mean
        honest_mean = {
            k: np.mean([w[k] for w in honest_weights], axis=0)
            for k in honest_weights[0].keys()
        }
        distance = np.linalg.norm(
            np.concatenate([
                (malicious[k] - honest_mean[k]).flatten()
                for k in honest_mean.keys()
            ])
        )
        
        print(f"  {name:20s}: distance from honest mean = {distance:.4f}")


def example_torch_legacy_interface():
    """Demonstrate legacy torch interface for backward compatibility."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Legacy Torch Interface")
    print("=" * 80)
    
    # Create torch tensors
    num_vectors = 10
    dim = 1000
    vectors = torch.randn(num_vectors, dim) * 0.1
    
    print(f"\nInput: {num_vectors} vectors of dimension {dim}")
    
    # MinMax (torch)
    print("\n--- MinMax (torch) ---")
    from decen_learn.attacks.minmax import craft_malicious_vector
    
    v_m, gamma, perturb_dir, v_ref = craft_malicious_vector(
        vectors=vectors,
        consensus_type="mean",
        oracle_type="minmax",
        gamma_init=20.0,
        tau=1e-3,
        perturb_kind="auto",
    )
    print(f"✓ Crafted malicious vector")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  Distance from mean: {torch.norm(v_m - v_ref).item():.4f}")
    
    # IPM (torch)
    print("\n--- IPM (torch) ---")
    from decen_learn.attacks.ipm import craft_ipm_local
    
    good_mask = [True] * (num_vectors - 2) + [False, False]  # 2 Byzantine
    v_m_ipm = craft_ipm_local(
        vectors=vectors,
        good_mask=good_mask,
        eps=0.5,
    )
    print(f"✓ Crafted malicious vector")
    print(f"  Distance from mean: {torch.norm(v_m_ipm - vectors.mean(dim=0)).item():.4f}")
    
    # LIE (torch)
    print("\n--- LIE (torch) ---")
    from decen_learn.attacks.lie import craft_malicious_vector as craft_lie
    
    v_m_lie = craft_lie(
        vectors=vectors,
        z=1.5,
        n=12,
        m=2,
    )
    print(f"✓ Crafted malicious vector")
    print(f"  Distance from mean: {torch.norm(v_m_lie - vectors.mean(dim=0)).item():.4f}")


def example_attack_comparison():
    """Compare attack effectiveness against different aggregators."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Attack vs Aggregator Comparison")
    print("=" * 80)
    
    # Setup
    num_honest = 7
    num_byzantine = 3
    total = num_honest + num_byzantine
    
    honest_weights = [create_mock_weights() for _ in range(num_honest)]
    attacker_weights = create_mock_weights()
    
    # Create attacks
    attacks = {
        "MinMax": MinMaxAttack(boosting_factor=1.0),
        "IPM": IPMAttack(eps=0.5),
        "LIE": LIEAttack(num_byzantine=num_byzantine, num_total=total),
    }
    
    print(f"\nSetup: {num_honest} honest + {num_byzantine} Byzantine = {total} total")
    print("\nSimulating attacks...")
    
    # Simulate aggregators (simplified)
    for attack_name, attack in attacks.items():
        print(f"\n--- {attack_name} Attack ---")
        
        # Craft malicious weights
        malicious = attack.craft(honest_weights, attacker_weights)
        
        # Simulate aggregation
        # 1. Mean aggregator (vulnerable)
        all_weights = honest_weights + [malicious] * num_byzantine
        mean_result = {
            k: np.mean([w[k] for w in all_weights], axis=0)
            for k in honest_weights[0].keys()
        }
        
        # 2. Honest mean (ground truth)
        honest_mean = {
            k: np.mean([w[k] for w in honest_weights], axis=0)
            for k in honest_weights[0].keys()
        }
        
        # Compute attack success (distance shift)
        shift = np.linalg.norm(
            np.concatenate([
                (mean_result[k] - honest_mean[k]).flatten()
                for k in honest_mean.keys()
            ])
        )
        
        print(f"  Mean aggregator shift: {shift:.4f}")
        print(f"  Attack success rate: {min(100, shift * 10):.1f}%")


def main():
    """Run all examples."""
    print("\n" + "🔴" * 40)
    print("BYZANTINE ATTACK MODULES - USAGE EXAMPLES")
    print("🔴" * 40)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run examples
    example_basic_usage()
    example_parameter_tuning()
    example_adversarial_scenario()
    example_torch_legacy_interface()
    example_attack_comparison()
    
    print("\n" + "=" * 80)
    print("✓ All examples completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Integrate attacks with ByzantineNode class")
    print("  2. Run full decentralized training simulation")
    print("  3. Evaluate attack effectiveness against different aggregators")
    print("  4. Implement adaptive attack strategies")
    print()


if __name__ == "__main__":
    main()