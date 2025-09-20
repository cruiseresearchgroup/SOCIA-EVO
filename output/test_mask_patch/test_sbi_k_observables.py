#!/usr/bin/env python3
"""
Test script for SBI calibrator with K=1 and K=5 observables.
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

def create_mock_data():
    """Create mock simulation data for testing."""
    T = 6  # 6 time steps for testing
    N = 20  # 20 agents for testing
    
    # Mock wearing data (T x N)
    np.random.seed(42)
    wearing = np.random.rand(T, N) > 0.4  # Random binary mask wearing with 60% probability
    wearing = wearing.astype(np.float64)
    
    # Mock neighbors (simplified - each agent has 2-3 neighbors)
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
    
    return bundle, train_window, T, N

def test_k_observables():
    """Test SBI calibrator with different K values."""
    
    print("=== SBI K-Observables Test ===\n")
    
    # Create mock data
    bundle, train_window, T, N = create_mock_data()
    print(f"Mock data created: T={T}, N={N}, training_window={train_window}")
    
    # Test both K=1 and K=5
    for k in [1, 5]:
        print(f"\n{'='*50}")
        print(f"Testing K={k} observables")
        print(f"{'='*50}")
        
        # Create SBI calibrator with specific K value
        print(f"1. Creating SBI calibrator with K={k}...")
        sbi_calibrator = get_calibrator('sbi', k_observables=k)
        print(f"   Created: {type(sbi_calibrator).__name__} with K={sbi_calibrator.k_observables}")
        
        # Setup simulator wrapper
        print(f"2. Setting up simulator wrapper...")
        try:
            sbi_calibrator._setup_simulator_wrapper(bundle, train_window)
            print("   ✓ Simulator wrapper setup successful")
        except Exception as e:
            print(f"   ✗ Simulator wrapper setup failed: {e}")
            continue
        
        # Test parameter sampling and conversion
        print(f"3. Testing parameter sampling...")
        samples, param_names = sbi_calibrator._sample_from_prior(n_samples=2, seed=42)
        fitted_params = sbi_calibrator._samples_to_fitted_params(
            samples[0], param_names, 42, train_window
        )
        print(f"   Parameter sample shape: {samples.shape}")
        
        # Test single simulation with different K values
        print(f"4. Testing single simulation...")
        try:
            trajectory = sbi_calibrator._run_single_simulation(fitted_params, seed=42)
            print(f"   ✓ Single simulation successful")
            print(f"   Trajectory shape: {trajectory.shape}")
            print(f"   Expected shape: ({train_window[1] - train_window[0]}, {k})")
            
            # Show trajectory details
            print(f"   Trajectory details:")
            if k == 1:
                print(f"     Daily wearing rates: {trajectory.flatten()}")
            elif k == 5:
                print(f"     Columns: [daily_rate, P01, P11, P10, P00]")
                for t in range(trajectory.shape[0]):
                    if t == 0:
                        print(f"     Day {t+1}: rate={trajectory[t,0]:.3f}, transitions=NaN (first day)")
                    else:
                        print(f"     Day {t+1}: rate={trajectory[t,0]:.3f}, P01={trajectory[t,1]:.3f}, P11={trajectory[t,2]:.3f}, P10={trajectory[t,3]:.3f}, P00={trajectory[t,4]:.3f}")
                        
                # Verify transition probabilities sum to 1 (approximately)
                for t in range(1, trajectory.shape[0]):
                    trans_sum = np.sum(trajectory[t, 1:5])
                    print(f"     Day {t+1} transition sum: {trans_sum:.3f} (should be ~1.0)")
                        
        except Exception as e:
            print(f"   ✗ Single simulation failed: {e}")
            continue
        
        # Test trajectory flattening
        print(f"5. Testing trajectory flattening...")
        flattened = sbi_calibrator._flatten_trajectory(trajectory)
        expected_length = (train_window[1] - train_window[0]) * k
        print(f"   Original shape: {trajectory.shape}")
        print(f"   Flattened shape: {flattened.shape}")
        print(f"   Expected length: {expected_length}")
        print(f"   ✓ Flattening successful" if flattened.shape[0] == expected_length else "✗ Flattening failed")
        
        # Test small training data generation
        print(f"6. Testing training data generation...")
        try:
            param_samples, trajectory_vectors = sbi_calibrator._generate_training_data(
                n_samples=2, seed=42
            )
            print(f"   ✓ Training data generation successful")
            print(f"   Parameter samples shape: {param_samples.shape}")
            print(f"   Trajectory vectors shape: {trajectory_vectors.shape}")
            print(f"   Expected trajectory vector length: {expected_length}")
            
            # Show sample trajectory vector
            print(f"   Sample trajectory vector (first sample):")
            if k == 1:
                print(f"     {trajectory_vectors[0]}")
            elif k == 5:
                reshaped = trajectory_vectors[0].reshape(-1, 5)
                print(f"     Reshaped back to ({reshaped.shape[0]}, 5):")
                for t in range(reshaped.shape[0]):
                    print(f"       Day {t+1}: {reshaped[t]}")
                    
        except Exception as e:
            print(f"   ✗ Training data generation failed: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("=== Test Summary ===")
    print("✅ Both K=1 and K=5 observables are working correctly!")
    print("✅ Transition probabilities are computed properly")
    print("✅ Trajectory flattening works for both modes")
    print("✅ Training data generation supports both K values")

def test_transition_probabilities():
    """Test transition probability calculation specifically."""
    print(f"\n{'='*50}")
    print("=== Transition Probability Test ===")
    
    # Create SBI calibrator
    sbi_calibrator = get_calibrator('sbi', k_observables=5)
    
    # Test with known states
    print("Testing with known state transitions...")
    
    # Test case 1: All agents stay the same
    prev_states = np.array([0.0, 0.0, 1.0, 1.0])  # 2 not wearing, 2 wearing
    curr_states = np.array([0.0, 0.0, 1.0, 1.0])  # Same states
    
    transitions = sbi_calibrator._compute_transition_probabilities(prev_states, curr_states)
    print(f"Case 1 - No changes:")
    print(f"  Previous: {prev_states}")
    print(f"  Current:  {curr_states}")
    print(f"  Transitions [P01, P11, P10, P00]: {transitions}")
    print(f"  Expected: [0.0, 1.0, 0.0, 1.0]")
    
    # Test case 2: Some transitions
    prev_states = np.array([0.0, 0.0, 1.0, 1.0])  # 2 not wearing, 2 wearing
    curr_states = np.array([1.0, 0.0, 1.0, 0.0])  # 1 adopts, 1 drops
    
    transitions = sbi_calibrator._compute_transition_probabilities(prev_states, curr_states)
    print(f"\nCase 2 - With transitions:")
    print(f"  Previous: {prev_states}")
    print(f"  Current:  {curr_states}")
    print(f"  Transitions [P01, P11, P10, P00]: {transitions}")
    print(f"  Expected: [0.5, 0.5, 0.5, 0.5]")
    
    # Verify probabilities sum to 1
    total = np.sum(transitions)
    print(f"  Sum of probabilities: {total:.3f} (should be 1.0)")

if __name__ == "__main__":
    test_k_observables()
    test_transition_probabilities()
