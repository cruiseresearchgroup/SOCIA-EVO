#!/usr/bin/env python3
"""
M3芯片性能测试脚本 - 测试不同SBI配置的性能和内存使用
"""

import sys
import os
import time
import psutil
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_code_using_calibration_template import (
    get_calibrator, 
    SimulationConfig,
    set_global_seed
)

def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_test_data():
    """创建测试数据"""
    T = 6
    N = 50  # 增加智能体数量来更好地测试性能
    
    np.random.seed(42)
    wearing = np.random.rand(T, N) > 0.4
    wearing = wearing.astype(np.float64)
    
    neighbors = {
        'family': [np.array([1, 2]) if i == 0 else np.array([0, (i+1) % N]) for i in range(N)],
        'work_school': [np.array([2, 3]) if i == 0 else np.array([0, (i+2) % N]) for i in range(N)],
        'community': [np.array([3, 4]) if i == 0 else np.array([0, (i+3) % N]) for i in range(N)]
    }
    
    risk = np.random.rand(N)
    age_oh = np.random.rand(N, 3)
    occ_oh = np.random.rand(N, 3)
    cfg = SimulationConfig()
    
    bundle = (wearing, neighbors, risk, age_oh, occ_oh, cfg)
    train_window = (1, T-2)
    
    return bundle, train_window

def test_sbi_config(config_name, k_observables, n_simulations, neural_net_config):
    """测试特定SBI配置的性能"""
    
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}")
    print(f"K observables: {k_observables}")
    print(f"N simulations: {n_simulations}")
    print(f"Neural config: {neural_net_config}")
    
    # 记录初始内存
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        # 创建SBI校准器
        print(f"\n1. 创建SBI校准器...")
        sbi_calibrator = get_calibrator('sbi', 
                                       k_observables=k_observables,
                                       n_simulations=n_simulations,
                                       neural_net_config=neural_net_config)
        
        creation_memory = get_memory_usage()
        print(f"   内存使用: {creation_memory - initial_memory:.1f} MB")
        
        # 设置仿真器
        bundle, train_window = create_test_data()
        sbi_calibrator._setup_simulator_wrapper(bundle, train_window)
        
        setup_memory = get_memory_usage()
        print(f"   设置后内存: {setup_memory - initial_memory:.1f} MB")
        
        # 生成训练数据
        print(f"\n2. 生成训练数据...")
        training_start = time.time()
        
        param_samples, trajectory_vectors = sbi_calibrator._generate_training_data(
            n_samples=n_simulations, seed=42
        )
        
        training_time = time.time() - training_start
        training_memory = get_memory_usage()
        
        print(f"   训练数据生成时间: {training_time:.1f} 秒")
        print(f"   训练数据内存: {training_memory - setup_memory:.1f} MB")
        print(f"   数据形状: params {param_samples.shape}, trajectories {trajectory_vectors.shape}")
        
        # 测试神经网络训练（占位符）
        print(f"\n3. 测试神经网络训练...")
        nn_start = time.time()
        
        sbi_calibrator._train_neural_posterior_estimator(param_samples, trajectory_vectors)
        
        nn_time = time.time() - nn_start
        nn_memory = get_memory_usage()
        
        print(f"   神经网络训练时间: {nn_time:.1f} 秒")
        print(f"   训练后内存: {nn_memory - training_memory:.1f} MB")
        
        # 总体性能
        total_time = time.time() - start_time
        total_memory = nn_memory - initial_memory
        
        print(f"\n4. 性能总结:")
        print(f"   总运行时间: {total_time:.1f} 秒")
        print(f"   总内存增长: {total_memory:.1f} MB")
        print(f"   平均每个样本时间: {total_time/n_simulations:.3f} 秒")
        
        # 性能评级
        if total_time < 30 and total_memory < 500:
            rating = "✅ 优秀"
        elif total_time < 60 and total_memory < 1000:
            rating = "✅ 良好"
        elif total_time < 120 and total_memory < 2000:
            rating = "⚠️ 可接受"
        else:
            rating = "❌ 需要优化"
            
        print(f"   M3芯片性能评级: {rating}")
        
        return True, total_time, total_memory
        
    except Exception as e:
        print(f"   ❌ 配置测试失败: {e}")
        return False, 0, 0

def main():
    """主测试函数"""
    print("=== M3芯片SBI性能测试 ===")
    print(f"系统信息:")
    print(f"  CPU: Apple M3")
    print(f"  内存: 16GB")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  初始内存使用: {get_memory_usage():.1f} MB")
    
    # 测试配置列表
    test_configs = [
        {
            'name': 'Ultra Minimal (调试用)',
            'k_observables': 1,
            'n_simulations': 20,
            'neural_net_config': {
                'batch_size': 8,
                'hidden_features': 8,
                'max_epochs': 5,
                'num_transforms': 2
            }
        },
        {
            'name': 'Light (快速实验)',
            'k_observables': 1,
            'n_simulations': 100,
            'neural_net_config': {
                'batch_size': 32,
                'hidden_features': 24,
                'max_epochs': 20,
                'num_transforms': 3
            }
        },
        {
            'name': 'Standard M3 (推荐)',
            'k_observables': 5,
            'n_simulations': 300,
            'neural_net_config': {
                'batch_size': 64,
                'hidden_features': 32,
                'max_epochs': 40,
                'num_transforms': 4
            }
        }
    ]
    
    results = []
    
    # 测试每个配置
    for config in test_configs:
        success, time_taken, memory_used = test_sbi_config(
            config['name'],
            config['k_observables'],
            config['n_simulations'],
            config['neural_net_config']
        )
        
        results.append({
            'name': config['name'],
            'success': success,
            'time': time_taken,
            'memory': memory_used,
            'n_simulations': config['n_simulations'],
            'k_observables': config['k_observables']
        })
    
    # 性能总结
    print(f"\n{'='*80}")
    print("=== M3芯片性能测试总结 ===")
    print(f"{'='*80}")
    
    print(f"{'配置':<20} {'成功':<6} {'时间(秒)':<10} {'内存(MB)':<10} {'样本数':<8} {'K值':<4}")
    print(f"{'-'*80}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{result['name']:<20} {status:<6} {result['time']:<10.1f} {result['memory']:<10.1f} "
              f"{result['n_simulations']:<8} {result['k_observables']:<4}")
    
    # 推荐配置
    print(f"\n推荐配置 (基于测试结果):")
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # 找到最佳平衡点
        best_config = min(successful_results, key=lambda x: x['time'] + x['memory']/100)
        print(f"  最佳配置: {best_config['name']}")
        print(f"  性能: {best_config['time']:.1f}秒, {best_config['memory']:.1f}MB")
        
        print(f"\n具体使用方法:")
        if best_config['name'] == 'Ultra Minimal (调试用)':
            print(f"  calibrator = get_calibrator('sbi', k_observables=1, n_simulations=20,")
            print(f"                             neural_net_config={{'batch_size': 8, 'hidden_features': 8}})")
        elif best_config['name'] == 'Light (快速实验)':
            print(f"  calibrator = get_calibrator('sbi', k_observables=1, n_simulations=100,")
            print(f"                             neural_net_config={{'batch_size': 32, 'hidden_features': 24}})")
        elif best_config['name'] == 'Standard M3 (推荐)':
            print(f"  calibrator = get_calibrator('sbi', k_observables=5, n_simulations=300,")
            print(f"                             neural_net_config={{'batch_size': 64, 'hidden_features': 32}})")
    
    print(f"\n注意事项:")
    print(f"  - M3芯片的统一内存架构很适合SBI训练")
    print(f"  - 建议从小配置开始，逐步增加复杂度")
    print(f"  - K=5提供更丰富信息，但内存需求更高")
    print(f"  - 实际性能可能因数据复杂度而异")

if __name__ == "__main__":
    main()
