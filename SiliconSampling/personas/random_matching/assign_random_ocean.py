"""
Random OCEAN Score Assignment Strategy

This script assigns random OCEAN personality scores (0-10 scale) to US Census personas.
This serves as a baseline comparison strategy to evaluate whether demographic-based 
matching (WPP, IPIP) produces more human-like personas than random assignment.

Input:
    - acs_pums_2023_labeled.parquet: Labeled census data with demographics
    
Output:
    - census_random_ocean_1m.parquet: Census data with random OCEAN scores (0-10)
    - census_random_ocean_preview.csv: Preview of first 1000 records
    
OCEAN Score Range: 0-10 (continuous uniform distribution)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define paths
SCRIPT_DIR = Path(__file__).parent
DATA_PREP_DIR = SCRIPT_DIR.parent.parent / "data_prep"
SYNTHETIC_CENSUS_PATH = DATA_PREP_DIR / "census_synthetic_1m.parquet"
OUTPUT_PARQUET = SCRIPT_DIR / "census_random_ocean_1m.parquet"
OUTPUT_PREVIEW = SCRIPT_DIR / "census_random_ocean_preview.csv"

# Target sample size
SAMPLE_SIZE = 1_000_000


def load_census_data():
    """Load shared synthetic census population"""
    print(f"Loading shared synthetic census from {SYNTHETIC_CENSUS_PATH}...")
    df = pd.read_parquet(SYNTHETIC_CENSUS_PATH)
    print(f"Loaded {len(df):,} records")
    return df


def assign_random_ocean_scores(df):
    """
    Assign random OCEAN scores (0-10) to each person in the synthetic population
    
    Args:
        df: Synthetic census DataFrame
        
    Returns:
        DataFrame with OCEAN scores added
    """
    print(f"\nAssigning random OCEAN scores to {len(df):,} personas...")
    
    df = df.copy()
    n_personas = len(df)
    
    # Assign random OCEAN scores (uniform distribution 0-10)
    df['Openness'] = np.random.uniform(0, 10, n_personas).round(2)
    df['Conscientiousness'] = np.random.uniform(0, 10, n_personas).round(2)
    df['Extraversion'] = np.random.uniform(0, 10, n_personas).round(2)
    df['Agreeableness'] = np.random.uniform(0, 10, n_personas).round(2)
    df['Neuroticism'] = np.random.uniform(0, 10, n_personas).round(2)
    
    print(f"✓ Assigned random OCEAN scores to {n_personas:,} personas")
    
    return df


def validate_random_ocean(df):
    """Validate the random OCEAN score distribution"""
    print("\n" + "="*70)
    print("RANDOM OCEAN SCORE VALIDATION")
    print("="*70)
    
    ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    print("\nOCEAN Score Statistics (0-10 scale):")
    print("-" * 70)
    for trait in ocean_traits:
        mean_score = df[trait].mean()
        std_score = df[trait].std()
        min_score = df[trait].min()
        max_score = df[trait].max()
        print(f"{trait:20s}: Mean={mean_score:5.2f}, Std={std_score:5.2f}, Range=[{min_score:.2f}, {max_score:.2f}]")
    
    print("\nExpected for uniform distribution: Mean≈5.0, Std≈2.89")
    print("="*70)


def save_output(df):
    """Save the dataset with random OCEAN scores"""
    print(f"\nSaving output files...")
    
    # Save full dataset as parquet
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✓ Saved full dataset: {OUTPUT_PARQUET} ({len(df):,} records)")
    
    # Save preview as CSV
    preview_df = df.head(1000)
    preview_df.to_csv(OUTPUT_PREVIEW, index=False)
    print(f"✓ Saved preview: {OUTPUT_PREVIEW} (1,000 records)")


def main():
    """Main execution pipeline"""
    print("="*70)
    print("RANDOM OCEAN SCORE ASSIGNMENT STRATEGY")
    print("="*70)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target sample size: {SAMPLE_SIZE:,}")
    print(f"OCEAN score range: 0-10 (continuous)")
    
    # Load data
    census_df = load_census_data()
    
    # Assign random OCEAN scores
    final_df = assign_random_ocean_scores(census_df)
    
    # Validate
    validate_random_ocean(final_df)
    
    # Save
    save_output(final_df)
    
    print("\n" + "="*70)
    print("RANDOM OCEAN ASSIGNMENT COMPLETE")
    print("="*70)
    print(f"Output: {OUTPUT_PARQUET}")
    print(f"Next step: Use this dataset to generate personas for validation testing")


if __name__ == "__main__":
    main()
