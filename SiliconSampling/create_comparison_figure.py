"""
Create Comparison Figure

This script creates a single comparison figure with 4 plots, where each plot
overlays data from 3 settings:
- v1 no persona
- v1 random deepseek
- v2 random deepseek

Output: results/comparison.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from agent_utils import (
    load_responses,
    load_ground_truth_data,
    extract_test_columns
)

from test.test_utils import (
    parse_synthetic_responses,
    calculate_all_metrics
)

# Set plotting style
sns.set_style("whitegrid")

# Define settings to compare
SETTINGS = [
    {
        'name': 'V1 No Persona',
        'path': 'responses_v1/no_persona/deepseek/responses_v1_no_persona_deepseek-chat.jsonl',
        'color': '#1f77b4',  # Blue
        'marker': 'o'
    },
    {
        'name': 'V1 Random',
        'path': 'responses_v1/random/deepseek/responses_v1_random_deepseek-chat.jsonl',
        'color': '#ff7f0e',  # Orange
        'marker': 's'
    },
    {
        'name': 'V2 Random',
        'path': 'responses_v2/random/deepseek/responses_v2_random_deepseek-chat.jsonl',
        'color': '#2ca02c',  # Green
        'marker': '^'
    }
]


def load_metrics_for_setting(response_file, ground_truth_df, test_cols):
    """Load responses and calculate metrics for a setting"""
    print(f"  Loading {response_file}...")
    
    # Load and parse responses
    responses = load_responses(response_file)
    synthetic_df = parse_synthetic_responses(responses)
    
    # Calculate metrics
    metrics = calculate_all_metrics(synthetic_df, test_cols)
    
    print(f"  ✓ Loaded {len(synthetic_df):,} responses")
    return metrics


def create_comparison_figure(all_metrics, settings, output_path):
    """Create comparison figure with all settings overlaid"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: MAE Distribution (overlaid histograms/KDE)
    ax = axes[0, 0]
    for i, (metrics, setting) in enumerate(zip(all_metrics, settings)):
        mae_values = list(metrics['mae_per_item'].values())
        ax.hist(mae_values, bins=20, alpha=0.5, label=f"{setting['name']}", 
                color=setting['color'], edgecolor='black', linewidth=0.5)
        ax.axvline(metrics['mae_overall'], color=setting['color'], linestyle='--', 
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Mean Absolute Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('MAE Distribution Across Items', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Synthetic vs Ground Truth Means (scatter plots)
    ax = axes[0, 1]
    for i, (metrics, setting) in enumerate(zip(all_metrics, settings)):
        ax.scatter(metrics['truth_means'], metrics['synth_means'], 
                  alpha=0.6, label=f"{setting['name']} (r={metrics['correlation']:.3f})",
                  color=setting['color'], marker=setting['marker'], s=60, edgecolors='black', linewidth=0.5)
    
    # Add perfect match line
    all_truth = np.concatenate([m['truth_means'] for m in all_metrics])
    min_val, max_val = min(all_truth), max(all_truth)
    ax.plot([min_val, max_val], [min_val, max_val], 
            'k--', linewidth=2, alpha=0.5, label='Perfect Match', zorder=0)
    
    ax.set_xlabel('Ground Truth Mean', fontsize=11)
    ax.set_ylabel('Synthetic Mean', fontsize=11)
    ax.set_title('Mean Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: K-S Test - Cumulative Distribution Functions
    ax = axes[1, 0]
    for i, (metrics, setting) in enumerate(zip(all_metrics, settings)):
        synth_sorted = np.sort(metrics['synth_means'])
        synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)
        ax.plot(synth_sorted, synth_cdf, 
                label=f"{setting['name']} (D={metrics['ks_statistic']:.4f})",
                linewidth=2.5, alpha=0.8, color=setting['color'])
    
    # Add ground truth (should be same for all, use first one)
    truth_sorted = np.sort(all_metrics[0]['truth_means'])
    truth_cdf = np.arange(1, len(truth_sorted) + 1) / len(truth_sorted)
    ax.plot(truth_sorted, truth_cdf, 
            label='Ground Truth', linewidth=3, alpha=0.9, color='black', linestyle=':')
    
    ax.set_xlabel('Question Mean Score', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('K-S Test: Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: K-S Test - CDF Differences
    ax = axes[1, 1]
    
    # Calculate CDF differences for each setting
    truth_sorted = np.sort(all_metrics[0]['truth_means'])
    
    for i, (metrics, setting) in enumerate(zip(all_metrics, settings)):
        synth_sorted = np.sort(metrics['synth_means'])
        
        # Create common grid
        all_means = np.sort(np.concatenate([synth_sorted, truth_sorted]))
        synth_cdf_interp = np.searchsorted(synth_sorted, all_means, side='right') / len(synth_sorted)
        truth_cdf_interp = np.searchsorted(truth_sorted, all_means, side='right') / len(truth_sorted)
        cdf_diff = np.abs(synth_cdf_interp - truth_cdf_interp)
        
        ax.plot(all_means, cdf_diff, color=setting['color'], linewidth=2.5, 
                alpha=0.8, label=f"{setting['name']} (Max={metrics['ks_statistic']:.4f})")
        ax.axhline(metrics['ks_statistic'], color=setting['color'], linestyle='--', 
                   linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Question Mean Score', fontsize=11)
    ax.set_ylabel('|CDF Difference|', fontsize=11)
    ax.set_title('K-S Test: Absolute CDF Difference', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # plt.suptitle('Validation Comparison: No Persona vs Random (V1 & V2)', 
                #  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comparison figure to {output_path}")


def main():
    """Main comparison pipeline"""
    print("\n" + "="*80)
    print("CREATING COMPARISON FIGURE")
    print("="*80 + "\n")
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth_df = load_ground_truth_data()
    test_cols = extract_test_columns(ground_truth_df)
    print(f"✓ Loaded ground truth: {test_cols.shape}\n")
    
    # Load metrics for each setting
    print("Loading metrics for each setting:")
    all_metrics = []
    
    for setting in SETTINGS:
        response_file = Path(setting['path'])
        if not response_file.exists():
            print(f"⚠️  Warning: {response_file} not found, skipping...")
            continue
        
        metrics = load_metrics_for_setting(response_file, ground_truth_df, test_cols)
        all_metrics.append(metrics)
    
    if len(all_metrics) == 0:
        print("\n❌ No valid response files found. Please generate responses first.")
        return
    
    # Create output directory
    result_dir = Path('results')
    result_dir.mkdir(exist_ok=True)
    output_path = result_dir / 'comparison.png'
    
    # Create comparison figure
    print("\nCreating comparison figure...")
    create_comparison_figure(all_metrics, SETTINGS[:len(all_metrics)], output_path)
    
    print("\n" + "="*80)
    print("✅ COMPARISON FIGURE COMPLETE")
    print("="*80)
    print(f"\nFigure saved to: {output_path.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
