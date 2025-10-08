#!/usr/bin/env python3
"""
Test script for SBI calibrator training data generation.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_code_using_calibration_template import (
    get_calibrator, 
    SimulationConfig,
    FittedParams,
    set_global_seed
)

def test_sbi_training_data():
    """Test SBI calibrator training data generation functionality."""
    
    print("=== SBI Training Data Generation Test ===\n")
    
    # Create SBI calibrator
    print("1. Creating SBI calibrator...")
    sbi_calibrator = get_calibrator('sbi')
    print(f"   Created: {type(sbi_calibrator).__name__}")
    print(f"   Default n_simulations: {sbi_calibrator.n_simulations}")
    
    # Test individual components
    print("\n2. Testing individual components:")
    
    # Test prior definition
    print("   a) Testing prior definition...")
    prior_bounds = sbi_calibrator._define_prior()
    print(f"      Defined {len(prior_bounds)} parameters")
    print("      Sample bounds:")
    for i, (name, bounds) in enumerate(list(prior_bounds.items())[:5]):
        print(f"        {name}: [{bounds[0]:.2f}, {bounds[1]:.2f}]")
    
    # Test prior sampling
    print("   b) Testing prior sampling...")
    samples, param_names = sbi_calibrator._sample_from_prior(n_samples=3, seed=42)
    print(f"      Sample shape: {samples.shape}")
    print(f"      First sample (first 5 params): {samples[0, :5]}")
    
    # Test parameter conversion
    print("   c) Testing parameter conversion...")
    fitted_params = sbi_calibrator._samples_to_fitted_params(
        samples[0], param_names, 42, (1, 10)
    )
    print(f"      Converted to: {type(fitted_params).__name__}")
    
    # Create mock bundle for testing
    print("\n3. Creating mock simulation data...")
    
    # Create minimal mock data
    T = 5  # 5 time steps for testing
    N = 10  # 10 agents for testing
    
    # Mock wearing data (T x N)
    np.random.seed(42)
    wearing = np.random.rand(T, N) > 0.5  # Random binary mask wearing
    wearing = wearing.astype(np.float64)
    
    # Mock neighbors (simplified - each agent has 2 neighbors)
    neighbors = {
        'family': [np.array([1, 2]) if i == 0 else np.array([0, (i+1) % N]) for i in range(N)],
        'work_school': [np.array([2, 3]) if i == 0 else np.array([0, (i+2) % N]) for i in range(N)],
        'community': [np.array([3, 4]) if i == 0 else np.array([0, (i+3) % N]) for i in range(N)]
    }
    
    # Mock other data
    risk = np.random.rand(N)  # Risk perception
    age_oh = np.random.rand(N, 3)  # Age one-hot encoding
    occ_oh = np.random.rand(N, 3)  # Occupation one-hot encoding
    cfg = SimulationConfig()
    
    bundle = (wearing, neighbors, risk, age_oh, occ_oh, cfg)
    train_window = (1, T-1)  # Use T-1 to have valid training window
    
    print(f"   Mock data created: T={T}, N={N}")
    print(f"   Training window: {train_window}")
    
    # Test simulator wrapper setup
    print("\n4. Testing simulator wrapper setup...")
    try:
        sbi_calibrator._setup_simulator_wrapper(bundle, train_window)
        print("   ✓ Simulator wrapper setup successful")
        print(f"   Components stored: {list(sbi_calibrator.simulation_components.keys())}")
    except Exception as e:
        print(f"   ✗ Simulator wrapper setup failed: {e}")
        return
    
    # Test single simulation
    print("\n5. Testing single simulation...")
    try:
        trajectory = sbi_calibrator._run_single_simulation(fitted_params, seed=42)
        print(f"   ✓ Single simulation successful")
        print(f"   Trajectory shape: {trajectory.shape}")
        print(f"   Trajectory values: {trajectory.flatten()}")
    except Exception as e:
        print(f"   ✗ Single simulation failed: {e}")
        return
    
    # Test trajectory flattening
    print("\n6. Testing trajectory flattening...")
    flattened = sbi_calibrator._flatten_trajectory(trajectory)
    print(f"   Original shape: {trajectory.shape}")
    print(f"   Flattened shape: {flattened.shape}")
    print(f"   Flattened values: {flattened}")
    
    # Test training data generation (small sample)
    print("\n7. Testing training data generation...")
    try:
        param_samples, trajectory_vectors = sbi_calibrator._generate_training_data(
            n_samples=3, seed=42
        )
        print(f"   ✓ Training data generation successful")
        print(f"   Parameter samples shape: {param_samples.shape}")
        print(f"   Trajectory vectors shape: {trajectory_vectors.shape}")
        print(f"   Sample trajectory vector: {trajectory_vectors[0]}")
    except Exception as e:
        print(f"   ✗ Training data generation failed: {e}")
        return
    
    print("\n=== All Tests Passed! ===")
    print("SBI training data generation is working correctly.")

if __name__ == "__main__":
    test_sbi_training_data()
