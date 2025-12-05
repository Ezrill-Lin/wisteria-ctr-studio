"""
IPIP-50 Demographic Matching Strategy

This script matches US Census personas to the IPIP-NEO Big Five dataset based on demographics,
then calculates OCEAN scores from the 50-item personality questionnaire responses.

Input:
    - ../wpp_matching/acs_pums_2023_labeled.parquet: Labeled census data
    - ../big5data.csv: IPIP-NEO dataset with demographic info and Big Five item responses
    
Output:
    - census_ipip_ocean_1m.parquet: Census data with IPIP-derived OCEAN scores (0-10 scale)
    - census_ipip_ocean_preview.csv: Preview of first 1000 records
    - ipip_matching_report.txt: Matching statistics and diagnostics
    
Methodology:
    1. Load IPIP Big Five dataset and census data
    2. Map IPIP demographics (race, age, gender) to standardized format
    3. Match each census persona to IPIP respondents with similar demographics
    4. Calculate OCEAN scores from IPIP-50 item responses (reverse-coded where needed)
    5. Scale scores from 1-5 to 0-10 range
    6. Fallback: If no demographic match, sample from full IPIP pool
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define paths
SCRIPT_DIR = Path(__file__).parent
DATA_PREP_DIR = SCRIPT_DIR.parent.parent / "data_prep"
SYNTHETIC_CENSUS_PATH = DATA_PREP_DIR / "census_synthetic_1m.parquet"
BIG5_DATA_PATH = SCRIPT_DIR.parent.parent / "big5data.csv"
OUTPUT_PARQUET = SCRIPT_DIR / "census_ipip_ocean_1m.parquet"
OUTPUT_PREVIEW = SCRIPT_DIR / "census_ipip_ocean_preview.csv"
MATCHING_REPORT = SCRIPT_DIR / "ipip_matching_report.txt"

# Target sample size
SAMPLE_SIZE = 1_000_000

# IPIP-50 item definitions
# Items ending with _R are reverse-coded (need to flip: 6 - score)
IPIP_COLUMNS = {
    'Extraversion': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'],
    'Neuroticism': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'],
    'Agreeableness': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'],
    'Conscientiousness': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
    'Openness': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']
}

# Reverse-coded items (based on IPIP-NEO codebook)
REVERSE_ITEMS = {
    'E2', 'E4', 'E6', 'E8', 'E10',  # Extraversion
    'N2', 'N4',                      # Neuroticism (only 2 reversed)
    'A1', 'A3', 'A5', 'A7',         # Agreeableness
    'C2', 'C4', 'C6', 'C8',         # Conscientiousness
    'O2', 'O4', 'O6'                 # Openness
}


def load_data():
    """Load census and IPIP datasets"""
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    # Load shared synthetic census
    print(f"\nLoading shared synthetic census from: {SYNTHETIC_CENSUS_PATH}")
    if not SYNTHETIC_CENSUS_PATH.exists():
        print(f"ERROR: Synthetic census file not found at {SYNTHETIC_CENSUS_PATH}")
        print("\nPlease run first: python generate_synthetic_census.py")
        sys.exit(1)
    
    census_df = pd.read_parquet(SYNTHETIC_CENSUS_PATH)
    print(f"✓ Synthetic census loaded: {len(census_df):,} records")
    print(f"  Columns: {list(census_df.columns)}")
    
    # Load IPIP Big Five data
    print(f"\nLoading IPIP data from: {BIG5_DATA_PATH}")
    if not BIG5_DATA_PATH.exists():
        print(f"ERROR: IPIP file not found at {BIG5_DATA_PATH}")
        sys.exit(1)
    
    ipip_df = pd.read_csv(BIG5_DATA_PATH, sep='\t')
    print(f"✓ IPIP data loaded: {len(ipip_df):,} respondents")
    print(f"  Columns: {len(ipip_df.columns)} total")
    
    # Filter out records with missing personality data
    personality_cols = []
    for trait_items in IPIP_COLUMNS.values():
        personality_cols.extend(trait_items)
    
    # Remove rows where all personality items are 0 (missed)
    ipip_df = ipip_df[ipip_df[personality_cols].sum(axis=1) > 0].copy()
    print(f"✓ After removing invalid responses: {len(ipip_df):,} respondents")
    
    return census_df, ipip_df


def reverse_code_item(score):
    """Reverse code an IPIP item (1-5 scale becomes 5-1)"""
    if pd.isna(score) or score == 0:  # 0 = missed
        return np.nan
    return 6 - score


def calculate_ocean_scores(ipip_df):
    """
    Calculate OCEAN scores from IPIP-50 item responses
    
    Process:
    1. Reverse-code items marked in REVERSE_ITEMS
    2. Calculate mean for each trait (10 items per trait)
    3. Scale from 1-5 to 0-10: (mean - 1) * 2.5
    
    Returns:
        DataFrame with OCEAN scores added
    """
    print("\n" + "="*80)
    print("CALCULATING OCEAN SCORES FROM IPIP-50 ITEMS")
    print("="*80)
    
    df = ipip_df.copy()
    
    # Apply reverse coding
    print("\nApplying reverse coding to items...")
    for item in REVERSE_ITEMS:
        if item in df.columns:
            df[item] = df[item].apply(reverse_code_item)
    
    # Calculate trait scores
    print("\nCalculating trait scores...")
    for trait, items in IPIP_COLUMNS.items():
        # Replace 0 (missed) with NaN for proper mean calculation
        item_data = df[items].replace(0, np.nan)
        
        # Calculate mean (ignoring NaN)
        trait_mean = item_data.mean(axis=1)
        
        # Scale from 1-5 to 0-10
        df[trait] = (trait_mean - 1) * 2.5
        
        # Show statistics
        valid_count = df[trait].notna().sum()
        mean_score = df[trait].mean()
        std_score = df[trait].std()
        print(f"  {trait:20s}: {valid_count:,} valid, Mean={mean_score:.2f}, Std={std_score:.2f}")
    
    return df


def map_ipip_demographics(ipip_df):
    """
    Map IPIP demographic codes to census-compatible format
    
    IPIP codes:
        race: Various codes (1-13, 0=missed)
        age: numeric (13+)
        gender: 1=Male, 2=Female, 3=Other, 0=missed
    """
    print("\n" + "="*80)
    print("MAPPING IPIP DEMOGRAPHICS")
    print("="*80)
    
    df = ipip_df.copy()
    
    # Age groups (align with common census age brackets)
    print("\nMapping age groups...")
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 17, 24, 34, 44, 54, 64, 200],
        labels=['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        right=True
    )
    
    # Gender mapping
    print("Mapping gender...")
    gender_map = {1: 'Male', 2: 'Female', 3: 'Other', 0: 'Unknown'}
    df['gender_mapped'] = df['gender'].map(gender_map).fillna('Unknown')
    
    # Create matching key (age_group|gender)
    df['matching_key'] = df['age_group'].astype(str) + '|' + df['gender_mapped'].astype(str)
    
    # Remove invalid records (under 18, missing demographics)
    df = df[df['age_group'] != 'Under 18'].copy()
    df = df[df['gender_mapped'] != 'Unknown'].copy()
    
    print(f"\n✓ Valid IPIP records for matching: {len(df):,}")
    print(f"✓ Unique demographic profiles: {df['matching_key'].nunique()}")
    
    # Show distribution
    print("\nAge group distribution:")
    print(df['age_group'].value_counts().sort_index())
    print("\nGender distribution:")
    print(df['gender_mapped'].value_counts())
    
    return df


def map_census_demographics(census_df):
    """
    Map census demographics to matching format
    """
    print("\n" + "="*80)
    print("MAPPING CENSUS DEMOGRAPHICS")
    print("="*80)
    
    df = census_df.copy()
    
    # Map age to age groups
    print("\nMapping census age to age groups...")
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 17, 24, 34, 44, 54, 64, 200],
        labels=['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        right=True
    )
    
    # Gender is already in correct format (Male/Female)
    df['gender_mapped'] = df['gender'].astype(str)
    
    # Create matching key
    df['matching_key'] = df['age_group'].astype(str) + '|' + df['gender_mapped'].astype(str)
    
    # Filter to adults only
    df = df[df['age_group'] != 'Under 18'].copy()
    
    print(f"\n✓ Census records for matching: {len(df):,}")
    print(f"✓ Unique demographic profiles: {df['matching_key'].nunique()}")
    
    return df


def match_and_assign_ocean(census_df, ipip_df):
    """
    Match census personas to IPIP respondents and assign OCEAN scores
    
    Strategy:
    1. Try to match on age_group + gender
    2. If no match, sample from full IPIP pool
    3. Randomly sample one IPIP respondent per census person
    """
    print("\n" + "="*80)
    print("MATCHING CENSUS TO IPIP AND ASSIGNING OCEAN SCORES")
    print("="*80)
    
    # Create IPIP pools by matching key
    ipip_pools = {}
    for key in ipip_df['matching_key'].unique():
        ipip_pools[key] = ipip_df[ipip_df['matching_key'] == key].index.tolist()
    
    print(f"\n✓ Created {len(ipip_pools)} IPIP demographic pools")
    
    # Initialize OCEAN score columns
    ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    for trait in ocean_traits:
        census_df[trait] = np.nan
    
    # Track matching statistics
    exact_matches = 0
    fallback_matches = 0
    
    print("\nMatching census personas to IPIP respondents...")
    print("(This may take a few minutes)")
    
    # Match in batches for progress tracking
    batch_size = 100000
    for batch_start in tqdm(range(0, len(census_df), batch_size), desc="Processing batches", ncols=100):
        batch_end = min(batch_start + batch_size, len(census_df))
        batch_indices = census_df.index[batch_start:batch_end]
        
        for idx in tqdm(batch_indices, desc=f"  Batch {batch_start//batch_size + 1}", leave=False, ncols=100):
            matching_key = census_df.loc[idx, 'matching_key']
            
            # Try exact match
            if matching_key in ipip_pools and len(ipip_pools[matching_key]) > 0:
                # Randomly sample from matching pool
                ipip_idx = np.random.choice(ipip_pools[matching_key])
                exact_matches += 1
            else:
                # Fallback: random sample from all IPIP
                ipip_idx = np.random.choice(ipip_df.index)
                fallback_matches += 1
            
            # Copy OCEAN scores
            for trait in ocean_traits:
                census_df.loc[idx, trait] = ipip_df.loc[ipip_idx, trait]
        
        # Progress update
        progress = (batch_end / len(census_df)) * 100
        print(f"  Progress: {batch_end:,}/{len(census_df):,} ({progress:.1f}%)")
    
    # Report matching statistics
    print("\n" + "="*80)
    print("MATCHING STATISTICS")
    print("="*80)
    print(f"Total census personas matched: {len(census_df):,}")
    print(f"Exact demographic matches: {exact_matches:,} ({exact_matches/len(census_df)*100:.1f}%)")
    print(f"Fallback (random) matches: {fallback_matches:,} ({fallback_matches/len(census_df)*100:.1f}%)")
    
    return census_df, exact_matches, fallback_matches


def validate_ocean_scores(df):
    """Validate the OCEAN score distribution"""
    print("\n" + "="*80)
    print("OCEAN SCORE VALIDATION")
    print("="*80)
    
    ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    print("\nOCEAN Score Statistics (0-10 scale):")
    print("-" * 80)
    for trait in ocean_traits:
        mean_score = df[trait].mean()
        std_score = df[trait].std()
        min_score = df[trait].min()
        max_score = df[trait].max()
        missing = df[trait].isna().sum()
        print(f"{trait:20s}: Mean={mean_score:5.2f}, Std={std_score:5.2f}, "
              f"Range=[{min_score:.2f}, {max_score:.2f}], Missing={missing}")
    
    print("\n" + "="*80)


def save_output(df, exact_matches, fallback_matches):
    """Save the dataset with IPIP-derived OCEAN scores"""
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    # Save full dataset as parquet
    print(f"\nSaving full dataset to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✓ Saved {len(df):,} records")
    
    # Save preview as CSV
    print(f"\nSaving preview (first 1000 rows) to: {OUTPUT_PREVIEW}")
    df.head(1000).to_csv(OUTPUT_PREVIEW, index=False)
    print("✓ Preview saved")
    
    # Save matching report
    print(f"\nSaving matching report to: {MATCHING_REPORT}")
    with open(MATCHING_REPORT, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IPIP-50 DEMOGRAPHIC MATCHING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Total personas: {len(df):,}\n\n")
        
        f.write("MATCHING STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Exact demographic matches: {exact_matches:,} ({exact_matches/len(df)*100:.1f}%)\n")
        f.write(f"Fallback (random) matches: {fallback_matches:,} ({fallback_matches/len(df)*100:.1f}%)\n\n")
        
        f.write("OCEAN SCORE STATISTICS (0-10 scale)\n")
        f.write("-"*80 + "\n")
        ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        for trait in ocean_traits:
            mean_score = df[trait].mean()
            std_score = df[trait].std()
            f.write(f"{trait:20s}: Mean={mean_score:5.2f}, Std={std_score:5.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print("✓ Report saved")
    print("\n" + "="*80)


def main():
    """Main execution function"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "IPIP-50 DEMOGRAPHIC MATCHING STRATEGY" + " "*21 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Step 1: Load data
    census_df, ipip_df = load_data()
    
    # Step 2: Calculate OCEAN scores from IPIP-50 items
    ipip_df = calculate_ocean_scores(ipip_df)
    
    # Step 3: Map demographics for matching
    ipip_df = map_ipip_demographics(ipip_df)
    census_df = map_census_demographics(census_df)
    
    # Note: We already have 1M from shared synthetic population, no need to sample
    print(f"\nUsing full shared synthetic population: {len(census_df):,} personas")
    
    # Step 5: Match and assign OCEAN scores
    census_df, exact_matches, fallback_matches = match_and_assign_ocean(census_df, ipip_df)
    
    # Step 6: Validate results
    validate_ocean_scores(census_df)
    
    # Step 7: Save output
    save_output(census_df, exact_matches, fallback_matches)
    
    print("\n✓ IPIP matching complete!")
    print(f"✓ Output: {OUTPUT_PARQUET}")
    print(f"✓ Preview: {OUTPUT_PREVIEW}")
    print(f"✓ Report: {MATCHING_REPORT}")
    print("\n")


if __name__ == "__main__":
    main()
