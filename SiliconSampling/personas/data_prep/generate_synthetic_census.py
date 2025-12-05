"""
Generate Shared Synthetic Census Population

This script generates a single 1M synthetic census population using stratified sampling
that will be used by ALL three OCEAN assignment strategies (WPP, Random, IPIP).

This ensures:
- Fair comparison between strategies (same demographic base)
- Reproducible results (same random seed)
- Privacy-preserving (stratified sampling with noise)

Input:
    - acs_pums_2023_labeled.parquet: Labeled census data
    
Output:
    - census_synthetic_1m.parquet: 1M synthetic census population (demographics only)
    - census_synthetic_preview.csv: Preview of first 1000 records

Usage:
    python generate_synthetic_census.py
    
This should be run ONCE before running any of the three OCEAN assignment strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the stratified sampling function
from synthesize_population import generate_synthetic_population

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# File paths
SCRIPT_DIR = Path(__file__).parent
LABELED_CENSUS_PATH = SCRIPT_DIR / "acs_pums_2023_labeled.parquet"
OUTPUT_PARQUET = SCRIPT_DIR / "census_synthetic_1m.parquet"
OUTPUT_PREVIEW = SCRIPT_DIR / "census_synthetic_preview.csv"

# Target sample size
SAMPLE_SIZE = 1_000_000


def main():
    """Generate shared synthetic census population."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "GENERATE SHARED SYNTHETIC CENSUS POPULATION" + " "*17 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Check if output already exists
    if OUTPUT_PARQUET.exists():
        print("="*80)
        print("WARNING: Synthetic population already exists!")
        print("="*80)
        print(f"File: {OUTPUT_PARQUET}")
        
        response = input("\nDo you want to regenerate? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nUsing existing synthetic population.")
            print("All three strategies will use this shared base.")
            return
        print("\nRegenerating synthetic population...")
    
    # Load labeled census data
    print("="*80)
    print("LOADING LABELED CENSUS DATA")
    print("="*80)
    
    if not LABELED_CENSUS_PATH.exists():
        print(f"ERROR: Census file not found: {LABELED_CENSUS_PATH}")
        print("\nPlease run: python label.py")
        sys.exit(1)
    
    census_df = pd.read_parquet(LABELED_CENSUS_PATH)
    print(f"✓ Loaded {len(census_df):,} census records")
    print(f"  Columns: {list(census_df.columns)}")
    
    # Generate synthetic population using stratified sampling
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC POPULATION VIA STRATIFIED SAMPLING")
    print("="*80)
    print(f"Target size: {SAMPLE_SIZE:,}")
    print(f"Random seed: {RANDOM_SEED}")
    print("\nThis preserves demographic distributions while ensuring privacy.")
    
    synthetic_df = generate_synthetic_population(
        census_df,
        target_n=SAMPLE_SIZE,
        seed=RANDOM_SEED
    )
    
    # Save output
    print("\n" + "="*80)
    print("SAVING SYNTHETIC POPULATION")
    print("="*80)
    
    print(f"\nSaving full dataset to: {OUTPUT_PARQUET}")
    synthetic_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✓ Saved {len(synthetic_df):,} records")
    
    print(f"\nSaving preview to: {OUTPUT_PREVIEW}")
    synthetic_df.head(1000).to_csv(OUTPUT_PREVIEW, index=False)
    print(f"✓ Saved 1,000 preview records")
    
    # Print summary
    print("\n" + "="*80)
    print("SYNTHETIC POPULATION SUMMARY")
    print("="*80)
    
    print(f"\nTotal records: {len(synthetic_df):,}")
    print(f"Columns: {len(synthetic_df.columns)}")
    
    print("\nDemographic distributions:")
    if 'gender' in synthetic_df.columns:
        print(f"\n  Gender:")
        for val, count in synthetic_df['gender'].value_counts().items():
            pct = count / len(synthetic_df) * 100
            print(f"    {val:20s}: {count:,} ({pct:.1f}%)")
    
    if 'age' in synthetic_df.columns:
        print(f"\n  Age statistics:")
        print(f"    Mean: {synthetic_df['age'].mean():.1f}")
        print(f"    Median: {synthetic_df['age'].median():.1f}")
        print(f"    Range: [{synthetic_df['age'].min()}, {synthetic_df['age'].max()}]")
    
    print("\n" + "="*80)
    print("✅ SHARED SYNTHETIC POPULATION GENERATED")
    print("="*80)
    print("\nThis dataset will be used by all three OCEAN assignment strategies:")
    print("  1. WPP Matching: wpp_matching/assign_wpp_ocean.py")
    print("  2. Random Assignment: random_matching/assign_random_ocean.py")
    print("  3. IPIP Matching: ipip_matching/match_ipip_demographics.py")
    print("\nNext steps:")
    print("  cd wpp_matching && python assign_wpp_ocean.py")
    print("  cd random_matching && python assign_random_ocean.py")
    print("  cd ipip_matching && python match_ipip_demographics.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
