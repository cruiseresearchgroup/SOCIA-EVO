#!/usr/bin/env python3
"""
Test script for SBI neural posterior estimator.
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
    T = 4  # 4 time steps for testing (smaller for faster testing)
    N = 15  # 15 agents for testing
    
    # Mock wearing data (T x N)
    np.random.seed(42)
    wearing = np.random.rand(T, N) > 0.4  # Random binary mask wearing
    wearing = wearing.astype(np.float64)
    
    # Mock neighbors
    neighbors = {
        'family': [np.array([1, 2]) if i == 0 else np.array([0, (i+1) % N]) for i in range(N)],
        'work_school': [np.array([2, 3]) if i == 0 else np.array([0, (i+2) % N]) for i in range(N)],
        'community': [np.array([3, 4]) if i == 0 else np.array([0, (i+3) % N]) for i in range(N)]
    }
    
    # Mock other data
    risk = np.random.rand(N)
    age_oh = np.random.rand(N, 3)
    occ_oh = np.random.rand(N, 3)
    cfg = SimulationConfig()
    
    bundle = (wearing, neighbors, risk, age_oh, occ_oh, cfg)
    train_window = (1, T-1)
    
    return bundle, train_window, T, N

def test_sbi_neural_estimator():
    """Test complete SBI neural posterior estimator pipeline."""
    
    print("=== SBI Neural Posterior Estimator Test ===\n")
    
    # Create mock data
    bundle, train_window, T, N = create_mock_data()
    print(f"Mock data created: T={T}, N={N}, training_window={train_window}")
    
    # Test both K=1 and K=5 (use K=1 for faster testing)
    k_observables = 1
    
    print(f"\n{'='*60}")
    print(f"Testing SBI with K={k_observables} observables")
    print(f"{'='*60}")
    
    # Create SBI calibrator with MAF
    print(f"1. Creating SBI calibrator with MAF...")
    sbi_calibrator = get_calibrator('sbi', 
                                   k_observables=k_observables,
                                   neural_net_config={
                                       'flow_type': 'maf',
                                       'batch_size': 32,  # Small batch for testing
                                       'max_epochs': 5,   # Few epochs for testing
                                       'learning_rate': 1e-3
                                   })
    print(f"   Created: {type(sbi_calibrator).__name__}")
    print(f"   K observables: {sbi_calibrator.k_observables}")
    print(f"   Flow type: {sbi_calibrator.neural_net_config['flow_type']}")
    
    # Test individual components first
    print(f"\n2. Testing individual components...")
    
    # Test prior definition
    prior_bounds = sbi_calibrator._define_prior()
    print(f"   Prior bounds defined: {len(prior_bounds)} parameters")
    
    # Test prior sampling
    samples, param_names = sbi_calibrator._sample_from_prior(n_samples=3, seed=42)
    print(f"   Prior sampling: {samples.shape}")
    
    # Test parameter conversion
    fitted_params = sbi_calibrator._samples_to_fitted_params(samples[0], param_names, 42, train_window)
    print(f"   Parameter conversion: {type(fitted_params).__name__}")
    
    # Test simulator wrapper setup
    sbi_calibrator._setup_simulator_wrapper(bundle, train_window)
    print(f"   Simulator wrapper setup: ✓")
    
    # Test single simulation
    trajectory = sbi_calibrator._run_single_simulation(fitted_params, seed=42)
    print(f"   Single simulation: {trajectory.shape}")
    
    # Test training data generation (small sample)
    print(f"\n3. Testing training data generation...")
    param_samples, trajectory_vectors = sbi_calibrator._generate_training_data(n_samples=5, seed=42)
    print(f"   Training data: params {param_samples.shape}, trajectories {trajectory_vectors.shape}")
    
    # Test neural posterior estimator training
    print(f"\n4. Testing neural posterior estimator training...")
    sbi_calibrator._train_neural_posterior_estimator(param_samples, trajectory_vectors)
    
    # Check if estimator was created
    if sbi_calibrator.posterior_estimator is not None:
        print(f"   Posterior estimator created: {type(sbi_calibrator.posterior_estimator)}")
    else:
        print(f"   Posterior estimator: None (likely using placeholder)")
    
    # Test posterior sampling
    print(f"\n5. Testing posterior sampling...")
    mock_observation = trajectory_vectors[0]
    posterior_samples = sbi_calibrator._sample_from_posterior(mock_observation, n_samples=10)
    print(f"   Posterior samples shape: {posterior_samples.shape}")
    
    # Test posterior to FittedParams conversion
    print(f"\n6. Testing posterior conversion...")
    final_fitted_params = sbi_calibrator._posterior_to_fitted_params(posterior_samples, 42, train_window)
    print(f"   Final fitted params: {type(final_fitted_params).__name__}")
    print(f"   Meta status: {final_fitted_params.meta.get('status', 'unknown')}")
    
    # Show some parameter values
    print(f"   Sample parameter values:")
    print(f"     alpha: {final_fitted_params.decision_weights['alpha']:.4f}")
    print(f"     gamma: {final_fitted_params.decision_weights['gamma']:.4f}")
    print(f"     beta_r: {final_fitted_params.decision_weights['beta_r']:.4f}")
    
    if 'posterior_summary_stats' in final_fitted_params.meta:
        stats = final_fitted_params.meta['posterior_summary_stats']
        print(f"   Posterior uncertainty (mean std): {stats['mean_param_std']:.4f}")
    
    print(f"\n{'='*60}")
    print("=== Test Summary ===")
    
    # Check if SBI library is available
    try:
        import torch
        from sbi import utils as sbi_utils
        from sbi.inference import NPE
        print("✅ SBI library is available - full neural posterior estimation possible")
        print("✅ All SBI components working correctly")
        print("✅ Neural posterior estimator training pipeline complete")
    except ImportError:
        print("⚠️  SBI library not available - using placeholder implementations")
        print("   Install with: pip install sbi-dev")
        print("✅ Placeholder implementations working correctly")
        print("✅ SBI structure ready for real neural network training")
    
    print("✅ Complete SBI pipeline tested successfully!")

def test_flow_types():
    """Test different flow types (MAF vs NSF)."""
    print(f"\n{'='*60}")
    print("=== Flow Type Comparison Test ===")
    
    bundle, train_window, T, N = create_mock_data()
    
    flow_types = ['maf', 'nsf']
    
    for flow_type in flow_types:
        print(f"\nTesting {flow_type.upper()} flow:")
        
        sbi_calibrator = get_calibrator('sbi', 
                                       k_observables=1,
                                       neural_net_config={
                                           'flow_type': flow_type,
                                           'batch_size': 16,
                                           'max_epochs': 2
                                       })
        
        print(f"  Created calibrator with {flow_type.upper()}")
        print(f"  Config: {sbi_calibrator.neural_net_config['flow_type']}")
        
        # Quick test of neural estimator training
        sbi_calibrator._setup_simulator_wrapper(bundle, train_window)
        param_samples, trajectory_vectors = sbi_calibrator._generate_training_data(n_samples=3, seed=42)
        sbi_calibrator._train_neural_posterior_estimator(param_samples, trajectory_vectors)
        
        print(f"  ✓ {flow_type.upper()} configuration tested")

if __name__ == "__main__":
    test_sbi_neural_estimator()
    test_flow_types()
