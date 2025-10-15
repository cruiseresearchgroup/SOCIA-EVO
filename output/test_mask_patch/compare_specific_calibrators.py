#!/usr/bin/env python3
"""
Compare specific calibrator results as requested by user.
Focus on the 7 specific test result folders.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

def load_metrics(file_path: str) -> Optional[Dict]:
    """Load test metrics from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_metric_values(metrics: Dict) -> Dict[str, Tuple[float, float]]:
    """
    Extract metric values and confidence intervals.
    
    Returns:
        Dictionary mapping metric names to (mean, std) tuples
    """
    result = {}
    
    # Define mapping from standard metric names to various possible JSON keys
    metric_mappings = {
        'rmse': ['rmse', 'RMSE_aggregate_mean', 'test_rmse_mean', 'rmse_mean'],
        'mae': ['mae', 'MAE_aggregate_mean', 'test_mae_mean', 'mae_mean'],
        'brier': ['brier', 'Brier_mean', 'test_brier_mean', 'brier_mean'],
        'transition_fit': ['transition_fit', 'TransitionFit_mean', 'test_transition_fit_mean', 'transition_fit_mean']
    }
    
    ci_mappings = {
        'rmse': ['rmse_std', 'RMSE_aggregate_CI95', 'test_rmse_std'],
        'mae': ['mae_std', 'MAE_aggregate_CI95', 'test_mae_std'],
        'brier': ['brier_std', 'Brier_CI95', 'test_brier_std'],
        'transition_fit': ['transition_fit_std', 'TransitionFit_CI95', 'test_transition_fit_std']
    }
    
    for metric in ['rmse', 'mae', 'brier', 'transition_fit']:
        mean_val = None
        std_val = None
        
        # Try to find mean value
        for key in metric_mappings[metric]:
            if key in metrics:
                if isinstance(metrics[key], dict) and 'mean' in metrics[key]:
                    mean_val = metrics[key]['mean']
                else:
                    mean_val = metrics[key]
                break
        
        # Try to find std/CI value
        for key in ci_mappings[metric]:
            if key in metrics:
                if isinstance(metrics[key], dict) and 'std' in metrics[key]:
                    std_val = metrics[key]['std']
                else:
                    # Assume CI95 is provided, convert to std (CI95 ≈ 1.96 * std)
                    std_val = metrics[key] / 1.96
                break
        
        if mean_val is not None and std_val is not None:
            result[metric] = (mean_val, std_val)
        elif mean_val is not None:
            # If no std found, use 0
            result[metric] = (mean_val, 0.0)
        else:
            print(f"Warning: {metric} not found in metrics")
    
    return result

def compare_specific_calibrators():
    """Compare the 7 specific calibrator results as requested."""
    print("=" * 80)
    print("🔍 比较指定的7种Calibrator测试结果")
    print("=" * 80)
    
    # Define the specific test result directories
    data_dir = "data_fitting/mask_adoption_data"
    specific_folders = {
        "BoCalibrator_TuRBO": "test_outputs_bo_TuRBO",
        "BoCalibrator_TuRBO_LLM_Guide": "test_outputs_bo_TuRBO_llm_guide", 
        "BoCalibrator_Vanilla": "test_outputs_bo_vanilla",
        "EvoCalibrator_GA": "test_outputs_evo_GA",
        "LogitHead": "test_outputs_logit_head",
        "RandomSearch": "test_outputs_random_search",
        "SBI": "test_outputs_sbi"
    }
    
    # Load metrics for each calibrator
    all_metrics = {}
    calibrator_files = {}
    
    for calibrator_name, folder_name in specific_folders.items():
        test_dir = os.path.join(data_dir, folder_name)
        metrics_file = os.path.join(test_dir, "test_metrics.json")
        
        if os.path.exists(metrics_file):
            print(f"✅ 找到 {calibrator_name}: {metrics_file}")
            metrics = load_metrics(metrics_file)
            if metrics:
                all_metrics[calibrator_name] = extract_metric_values(metrics)
                calibrator_files[calibrator_name] = metrics_file
        else:
            print(f"❌ 未找到 {calibrator_name}: {metrics_file}")
    
    if not all_metrics:
        print("❌ 无法加载任何有效的metrics数据")
        return
    
    print(f"\n✅ 成功加载 {len(all_metrics)} 个calibrator结果")
    
    print("\n" + "=" * 80)
    print("🏆 7种Calibrator性能全面比较分析")
    print("=" * 80)
    
    # Display detailed results
    print("\n📊 详细测试结果（均值 ± 95%置信区间）")
    print("-" * 80)
    
    metrics_order = ['rmse', 'mae', 'brier', 'transition_fit']
    metric_names = ['RMSE', 'MAE', 'Brier', 'TransitionFit']
    
    rankings = {}
    
    for i, (metric, display_name) in enumerate(zip(metrics_order, metric_names)):
        print(f"\n{display_name}（越小越好）:")
        
        # Get values for this metric from all calibrators
        metric_values = []
        for name in all_metrics:
            if metric in all_metrics[name]:
                mean_val, std_val = all_metrics[name][metric]
                metric_values.append((mean_val, name))
        
        # Sort by metric value (ascending for all metrics)
        metric_values.sort(key=lambda x: x[0])
        
        # Record rankings
        for rank, (value, name) in enumerate(metric_values):
            if name not in rankings:
                rankings[name] = []
            rankings[name].append(rank + 1)
        
        # Display results with emojis for top 3
        for rank, (value, name) in enumerate(metric_values):
            mean_val, std_val = all_metrics[name][metric]
            ci_95 = 1.96 * std_val  # 95% confidence interval
            
            if rank == 0:
                print(f"  🥇 {name:<30}: {mean_val:.4f} ± {ci_95:.4f}")
            elif rank == 1:
                print(f"  🥈 {name:<30}: {mean_val:.4f} ± {ci_95:.4f}")
            elif rank == 2:
                print(f"  🥉 {name:<30}: {mean_val:.4f} ± {ci_95:.4f}")
            else:
                print(f"     {name:<30}: {mean_val:.4f} ± {ci_95:.4f}")
    
    # Calculate average rankings
    print("\n🏅 综合排名分析")
    print("-" * 70)
    
    avg_rankings = []
    for name in rankings:
        avg_rank = np.mean(rankings[name])
        avg_rankings.append((avg_rank, name))
    
    avg_rankings.sort(key=lambda x: x[0])
    
    # Display rankings
    for i, (avg_rank, name) in enumerate(avg_rankings):
        rank_details = ", ".join([f"{metric_names[j]}={rankings[name][j]}" for j in range(len(rankings[name]))])
        
        if i == 0:
            print(f"🥇 第1名: {name:<30} (平均排名: {avg_rank:.1f})")
        elif i == 1:
            print(f"🥈 第2名: {name:<30} (平均排名: {avg_rank:.1f})")
        elif i == 2:
            print(f"🥉 第3名: {name:<30} (平均排名: {avg_rank:.1f})")
        else:
            print(f"   第{i+1}名: {name:<30} (平均排名: {avg_rank:.1f})")
        print(f"       各指标排名: {rank_details}")
    
    # Performance gap analysis
    if len(avg_rankings) > 1:
        best_name = avg_rankings[0][1]
        best_metrics = all_metrics[best_name]
        
        print(f"\n📈 相对于最佳方法({best_name})的性能差距")
        print("-" * 70)
        
        for avg_rank, name in avg_rankings[1:]:
            print(f"\n{name} vs {best_name}:")
            for metric in metrics_order:
                if metric in best_metrics and metric in all_metrics[name]:
                    best_val = best_metrics[metric][0]
                    current_val = all_metrics[name][metric][0]
                    
                    if best_val > 0:
                        gap_percent = ((current_val - best_val) / best_val) * 100
                        metric_display = metric.upper().replace('_', '')
                        print(f"  {metric_display}: {gap_percent:+.1f}% 差距 ({current_val:.4f} vs {best_val:.4f})")
    
    # Detailed analysis by algorithm type
    print(f"\n🔍 算法类型分析")
    print("-" * 70)
    
    algorithm_groups = {
        "贝叶斯优化": ["BoCalibrator_TuRBO", "BoCalibrator_TuRBO_LLM_Guide", "BoCalibrator_Vanilla"],
        "进化算法": ["EvoCalibrator_GA"],
        "传统方法": ["SBI", "RandomSearch", "LogitHead"]
    }
    
    for group_name, calibrators in algorithm_groups.items():
        print(f"\n{group_name}:")
        for calibrator in calibrators:
            if calibrator in all_metrics:
                avg_rank = np.mean(rankings[calibrator])
                print(f"  - {calibrator}: 平均排名 {avg_rank:.1f}")
    
    print(f"\n🎯 总结和建议:")
    print("-" * 70)
    if avg_rankings:
        best_calibrator = avg_rankings[0][1]
        print(f"✅ 推荐使用: {best_calibrator}")
        
        # Special notes for different calibrator types
        if "TuRBO" in best_calibrator:
            if "LLM_Guide" in best_calibrator:
                print("   💡 此版本使用了领域知识指导和风险感知的TuRBO优化")
            else:
                print("   💡 此版本使用了Trust Region贝叶斯优化")
        elif "Evo" in best_calibrator:
            print("   💡 此版本使用了遗传算法进行参数优化")
        elif best_calibrator == "SBI":
            print("   💡 SBI使用神经网络进行贝叶斯推断")
        elif best_calibrator == "RandomSearch":
            print("   💡 随机搜索作为基准方法")
        elif best_calibrator == "LogitHead":
            print("   💡 逻辑回归作为传统机器学习方法")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_specific_calibrators()
