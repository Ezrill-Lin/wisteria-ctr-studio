"""
Automated Validation Pipeline

This script automatically validates all existing response files and generates:
- Printed validation metrics (saved to text files)
- Visualization plots (saved as PNG files)

Output structure: results/{version}/{strategy}/{api_provider}/
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

from agent_utils import (
    load_responses,
    load_ground_truth_data,
    extract_test_columns,
    detect_api_provider
)

from test.test_utils import (
    parse_synthetic_responses,
    calculate_all_metrics,
    print_validation_results
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def capture_validation_output(metrics, label, num_synthetic, num_ground_truth):
    """Capture print_validation_results output to string"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    print_validation_results(metrics, label, num_synthetic, num_ground_truth)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return output


def create_validation_plots(metrics, label, output_path):
    """Create and save validation plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: MAE distribution
    mae_values = list(metrics['mae_per_item'].values())
    axes[0, 0].hist(mae_values, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(metrics['mae_overall'], color='red', linestyle='--', linewidth=2, 
                    label=f"Mean MAE: {metrics['mae_overall']:.4f}")
    axes[0, 0].set_xlabel('Mean Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MAE Distribution Across Items')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Synthetic vs Ground Truth Means
    axes[0, 1].scatter(metrics['truth_means'], metrics['synth_means'], alpha=0.6)
    axes[0, 1].plot([min(metrics['truth_means']), max(metrics['truth_means'])], 
                 [min(metrics['truth_means']), max(metrics['truth_means'])], 
                 'r--', linewidth=2, label='Perfect Match')
    axes[0, 1].set_xlabel('Ground Truth Mean')
    axes[0, 1].set_ylabel('Synthetic Mean')
    axes[0, 1].set_title(f"Mean Comparison (r={metrics['correlation']:.3f})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: K-S Test - Cumulative Distribution Functions
    synth_sorted = np.sort(metrics['synth_means'])
    truth_sorted = np.sort(metrics['truth_means'])
    synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)
    truth_cdf = np.arange(1, len(truth_sorted) + 1) / len(truth_sorted)
    
    axes[1, 0].plot(synth_sorted, synth_cdf, label='Synthetic', linewidth=2, alpha=0.8)
    axes[1, 0].plot(truth_sorted, truth_cdf, label='Ground Truth', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Question Mean Score')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title(f"K-S Test: Cumulative Distribution\nD={metrics['ks_statistic']:.4f}, p={metrics['ks_p_value']:.4f}")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: K-S Test - Difference between CDFs
    all_means = np.sort(np.concatenate([synth_sorted, truth_sorted]))
    synth_cdf_interp = np.searchsorted(synth_sorted, all_means, side='right') / len(synth_sorted)
    truth_cdf_interp = np.searchsorted(truth_sorted, all_means, side='right') / len(truth_sorted)
    cdf_diff = np.abs(synth_cdf_interp - truth_cdf_interp)
    
    axes[1, 1].fill_between(all_means, 0, cdf_diff, alpha=0.3, color='red')
    axes[1, 1].plot(all_means, cdf_diff, color='red', linewidth=2)
    axes[1, 1].axhline(metrics['ks_statistic'], color='black', linestyle='--', 
                       linewidth=2, label=f"Max Difference (D={metrics['ks_statistic']:.4f})")
    axes[1, 1].set_xlabel('Question Mean Score')
    axes[1, 1].set_ylabel('|CDF Difference|')
    axes[1, 1].set_title('K-S Test: Absolute CDF Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Validation Results - {label}", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def validate_response_file(response_file, ground_truth_df, test_cols, result_base_dir):
    """Validate a single response file and save results"""
    
    # Parse path to get version, strategy, api_provider, model
    parts = response_file.parts
    version_idx = [i for i, p in enumerate(parts) if p.startswith('responses_')][0]
    version = parts[version_idx].replace('responses_', '')
    strategy = parts[version_idx + 1]
    api_provider = parts[version_idx + 2]
    model = response_file.stem.replace(f'responses_{version}_{strategy}_', '')
    
    print(f"\n{'='*80}")
    print(f"Validating: {version}/{strategy}/{api_provider}/{model}")
    print(f"{'='*80}")
    
    try:
        # Load and parse responses
        responses = load_responses(response_file)
        print(f"✓ Loaded {len(responses)} responses")
        
        synthetic_df = parse_synthetic_responses(responses)
        print(f"✓ Parsed {len(synthetic_df):,} valid responses")
        
        if len(synthetic_df) == 0:
            print("⚠️  No valid responses to validate, skipping...")
            return
        
        # Calculate metrics
        metrics = calculate_all_metrics(synthetic_df, test_cols)
        print("✓ Calculated metrics")
        
        # Create output directory
        output_dir = result_base_dir / version / strategy / api_provider
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics text
        label = f"{strategy} ({model})"
        metrics_text = capture_validation_output(metrics, label, len(synthetic_df), len(test_cols))
        metrics_file = output_dir / f"metrics_{version}_{strategy}_{model}.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(metrics_text)
        print(f"✓ Saved metrics to {metrics_file}")
        
        # Save plots
        plot_file = output_dir / f"plots_{version}_{strategy}_{model}.png"
        create_validation_plots(metrics, label, plot_file)
        print(f"✓ Saved plots to {plot_file}")
        
        print(f"✅ Validation complete for {version}/{strategy}/{api_provider}/{model}")
        
    except Exception as e:
        print(f"❌ Error validating {response_file}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main validation pipeline"""
    print("\n" + "="*80)
    print("AUTOMATED VALIDATION PIPELINE")
    print("="*80 + "\n")
    
    # Load ground truth data once
    print("Loading ground truth data...")
    ground_truth_df = load_ground_truth_data()
    test_cols = extract_test_columns(ground_truth_df)
    print(f"✓ Loaded ground truth: {test_cols.shape}\n")
    
    # Find all response files
    response_files = []
    for version_dir in Path('.').glob('responses_*'):
        if version_dir.is_dir():
            response_files.extend(version_dir.glob('*/*/*.jsonl'))
    
    print(f"Found {len(response_files)} response files to validate:\n")
    for rf in response_files:
        print(f"  - {rf}")
    print()
    
    if len(response_files) == 0:
        print("⚠️  No response files found. Please generate responses first.")
        print("   Example: python agent.py --strategy random --model deepseek-chat --sample-size 2000")
        return
    
    # Create results base directory
    result_base_dir = Path('results')
    result_base_dir.mkdir(exist_ok=True)
    
    # Validate each file
    for i, response_file in enumerate(response_files, 1):
        print(f"\n[{i}/{len(response_files)}]")
        validate_response_file(response_file, ground_truth_df, test_cols, result_base_dir)
    
    print("\n" + "="*80)
    print("✅ VALIDATION PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {result_base_dir.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
