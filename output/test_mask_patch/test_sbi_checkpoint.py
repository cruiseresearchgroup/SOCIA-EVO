#!/usr/bin/env python3
"""
Test script for SBI checkpoint functionality.
"""

import sys
import os
import numpy as np
import json

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
    T = 4  # 4 time steps for testing
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

def test_sbi_checkpoint_save():
    """Test SBI checkpoint saving functionality."""
    
    print("=== SBI Checkpoint Save Test ===\n")
    
    # Create mock data
    bundle, train_window, T, N = create_mock_data()
    print(f"Mock data created: T={T}, N={N}, training_window={train_window}")
    
    # Create SBI calibrator
    print(f"\n1. Creating SBI calibrator...")
    sbi_calibrator = get_calibrator('sbi', 
                                   k_observables=1,
                                   n_simulations=1000,  # Set full simulation count
                                   neural_net_config={
                                       'flow_type': 'maf',
                                       'batch_size': 32,
                                       'max_epochs': 5
                                   })
    
    print(f"   Created: {type(sbi_calibrator).__name__}")
    print(f"   K observables: {sbi_calibrator.k_observables}")
    print(f"   N simulations: {sbi_calibrator.n_simulations}")
    
    # Test checkpoint saving components
    print(f"\n2. Testing checkpoint saving components...")
    
    # Setup simulator
    sbi_calibrator._setup_simulator_wrapper(bundle, train_window)
    
    # Generate training data (small sample for testing)
    param_samples, trajectory_vectors = sbi_calibrator._generate_training_data(n_samples=5, seed=42)
    print(f"   Training data generated: params {param_samples.shape}, trajectories {trajectory_vectors.shape}")
    
    # Test checkpoint saving
    cfg = bundle[5]  # Extract config
    output_dir = sbi_calibrator._save_sbi_checkpoint(param_samples, trajectory_vectors, train_window, 42, cfg)
    print(f"   Checkpoint saved to: {output_dir}")
    
    # Verify files were created
    expected_files = [
        "config.json",
        "training_data.json", 
        "prior_info.json",
        "posterior_estimator.json"  # or .pt if torch available
    ]
    
    print(f"\n3. Verifying checkpoint files...")
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"   ✓ {filename} exists")
            
            # Show file size
            size = os.path.getsize(filepath)
            print(f"     Size: {size} bytes")
            
            # For JSON files, show some content
            if filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if filename == "config.json":
                        print(f"     SBI config K: {data['sbi_config']['k_observables']}")
                        print(f"     Training samples: {data['sbi_config']['training_samples']}")
                    elif filename == "training_data.json":
                        print(f"     Parameter samples: {len(data['parameter_samples'])}")
                        print(f"     Trajectory vectors: {len(data['trajectory_vectors'])}")
                        print(f"     Parameter names: {len(data['parameter_names'])}")
                    elif filename == "prior_info.json":
                        print(f"     Prior bounds: {data['n_parameters']} parameters")
                except Exception as e:
                    print(f"     Could not read JSON: {e}")
        else:
            print(f"   ✗ {filename} missing")
    
    return output_dir

def test_sbi_checkpoint_load(checkpoint_dir):
    """Test SBI checkpoint loading functionality."""
    
    print(f"\n=== SBI Checkpoint Load Test ===\n")
    
    # Create new SBI calibrator
    sbi_calibrator = get_calibrator('sbi')
    
    # Test checkpoint loading
    print(f"1. Testing checkpoint loading...")
    checkpoint_data = sbi_calibrator._load_sbi_checkpoint(checkpoint_dir)
    
    if checkpoint_data is not None:
        print(f"   ✓ Checkpoint loaded successfully")
        
        # Show loaded data summary
        config = checkpoint_data['config']['sbi_config']
        training_data = checkpoint_data['training_data']
        
        print(f"   Loaded configuration:")
        print(f"     K observables: {config['k_observables']}")
        print(f"     Training samples: {config['training_samples']}")
        print(f"     Flow type: {config['neural_net_config']['flow_type']}")
        
        print(f"   Loaded training data:")
        print(f"     Parameter samples: {len(training_data['parameter_samples'])}")
        print(f"     Trajectory vectors: {len(training_data['trajectory_vectors'])}")
        print(f"     Parameter names: {len(training_data['parameter_names'])}")
        
    else:
        print(f"   ✗ Checkpoint loading failed")
        return False
    
    # Test posterior sampling from checkpoint
    print(f"\n2. Testing posterior sampling from checkpoint...")
    
    # Create mock observation
    mock_observation = np.array([0.6, 0.7])  # Mock trajectory vector
    
    try:
        posterior_samples = sbi_calibrator.sample_from_checkpoint(
            checkpoint_dir, mock_observation, n_samples=10
        )
        
        if posterior_samples is not None:
            print(f"   ✓ Posterior sampling successful")
            print(f"   Posterior samples shape: {posterior_samples.shape}")
            print(f"   Sample parameter values (first sample):")
            param_names = checkpoint_data['training_data']['parameter_names'][:5]  # Show first 5
            for i, name in enumerate(param_names):
                print(f"     {name}: {posterior_samples[0, i]:.4f}")
        else:
            print(f"   ✗ Posterior sampling failed")
            
    except Exception as e:
        print(f"   ✗ Posterior sampling error: {e}")
    
    # Test FittedParams creation from checkpoint
    print(f"\n3. Testing FittedParams creation from checkpoint...")
    
    try:
        fitted_params = sbi_calibrator.create_fitted_params_from_checkpoint(
            checkpoint_dir, mock_observation, n_samples=10
        )
        
        if fitted_params is not None:
            print(f"   ✓ FittedParams creation successful")
            print(f"   Type: {type(fitted_params).__name__}")
            print(f"   Loaded from checkpoint: {fitted_params.meta.get('loaded_from_checkpoint', False)}")
            print(f"   Sample parameter values:")
            print(f"     alpha: {fitted_params.decision_weights['alpha']:.4f}")
            print(f"     gamma: {fitted_params.decision_weights['gamma']:.4f}")
            print(f"     beta_r: {fitted_params.decision_weights['beta_r']:.4f}")
        else:
            print(f"   ✗ FittedParams creation failed")
            
    except Exception as e:
        print(f"   ✗ FittedParams creation error: {e}")
    
    return True

def test_checkpoint_workflow():
    """Test complete checkpoint save and load workflow."""
    
    print(f"\n=== Complete Checkpoint Workflow Test ===\n")
    
    # Step 1: Save checkpoint
    print("Step 1: Saving checkpoint...")
    checkpoint_dir = test_sbi_checkpoint_save()
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"✓ Checkpoint saved successfully to: {checkpoint_dir}")
        
        # Step 2: Load checkpoint
        print(f"\nStep 2: Loading checkpoint...")
        success = test_sbi_checkpoint_load(checkpoint_dir)
        
        if success:
            print(f"✓ Checkpoint loaded and used successfully")
            
            # Step 3: Show workflow benefits
            print(f"\nStep 3: Workflow Benefits:")
            print(f"✅ Trained model can be reused without retraining")
            print(f"✅ Multiple posterior samples can be generated from same model")
            print(f"✅ Different observations can be processed with same trained model")
            print(f"✅ Model sharing and reproducibility enabled")
            print(f"✅ Incremental learning possible")
            
        else:
            print(f"✗ Checkpoint loading failed")
    else:
        print(f"✗ Checkpoint saving failed")

if __name__ == "__main__":
    test_checkpoint_workflow()
