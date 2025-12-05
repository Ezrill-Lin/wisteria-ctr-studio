"""
WPP OCEAN Assignment Strategy

This script assigns OCEAN personality traits from Vietnamese WPP survey data to US Census personas.
It performs the complete WPP matching workflow in a single script.

Input:
    - ../acs_pums_2023_labeled.parquet: Labeled census data
    - wpp_merged_result_VT.xlsx: Vietnamese WPP survey data
    
Output:
    - census_with_personality_1m.parquet: 1M personas with WPP OCEAN levels (text-based)
    - census_with_personality_preview.csv: Preview of first 1000 records

Workflow:
    1. Load and apply WPP demographic mappings to census data
    2. Generate 1M synthetic population via stratified sampling
    3. Load WPP survey data and extract OCEAN traits
    4. Match census personas to WPP respondents based on demographics
    5. Assign OCEAN traits (fallback to random if no exact match)
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# File paths
SCRIPT_DIR = Path(__file__).parent
DATA_PREP_DIR = SCRIPT_DIR.parent.parent / "data_prep"
SYNTHETIC_CENSUS_PATH = DATA_PREP_DIR / "census_synthetic_1m.parquet"
WPP_SURVEY_PATH = SCRIPT_DIR / "wpp_merged_result_VT.xlsx"
OUTPUT_PARQUET = SCRIPT_DIR / "census_with_personality_1m.parquet"
OUTPUT_PREVIEW = SCRIPT_DIR / "census_with_personality_preview.csv"

# Target sample size
SAMPLE_SIZE = 1_000_000


# =============================================================================
# STEP 1: WPP DEMOGRAPHIC MAPPINGS
# =============================================================================

def map_age_to_wpp(age):
    """Map census age to WPP age groups."""
    if pd.isna(age) or age < 18:
        return None
    elif 18 <= age <= 25:
        return "18-25 (Gen Z)"
    elif 26 <= age <= 41:
        return "26-41 (Millennials)"
    elif 42 <= age <= 55:
        return "42-55 (Gen X)"
    else:
        return "56+ (Older)"


def map_employment_to_wpp(emp_status):
    """Map census employment to WPP employment categories."""
    mapping = {
        'Employed': 'Working full time',
        'Unemployed': 'Unemployed, in between jobs',
        'Not in labor force': 'Not in labor force'
    }
    return mapping.get(emp_status, emp_status)


def categorize_monthly_income(annual_income):
    """Categorize annual income into monthly bins for WPP matching."""
    if pd.isna(annual_income) or annual_income < 0:
        return "No income"
    
    monthly = annual_income / 12
    
    if monthly < 2000:
        return "$0 - $1,999"
    elif monthly < 4000:
        return "$2,000 - $3,999"
    elif monthly < 6000:
        return "$4,000 - $5,999"
    elif monthly < 10000:
        return "$6,000 - $9,999"
    else:
        return "$10,000 and above"


def apply_wpp_mappings(df):
    """Apply WPP demographic mappings to census data."""
    print("\n" + "="*80)
    print("STEP 1: APPLYING WPP DEMOGRAPHIC MAPPINGS")
    print("="*80)
    
    df = df.copy()
    df['wpp_age_group'] = df['age'].apply(map_age_to_wpp)
    df['wpp_gender'] = df['gender']
    df['wpp_employment'] = df['employment_status'].apply(map_employment_to_wpp)
    df['wpp_income_range'] = df['annual_income_usd'].apply(categorize_monthly_income)
    
    # Filter to valid age groups (18+)
    df = df[df['wpp_age_group'].notna()].copy()
    
    # Create matching key
    df['matching_key'] = (
        df['wpp_age_group'].astype(str) + '|' +
        df['wpp_gender'].astype(str) + '|' +
        df['wpp_employment'].astype(str) + '|' +
        df['wpp_income_range'].astype(str)
    )
    
    print(f"✓ Mapped {len(df):,} census records")
    print(f"✓ Unique demographic profiles: {df['matching_key'].nunique():,}")
    
    return df


# =============================================================================
# STEP 2: LOAD WPP SURVEY AND EXTRACT OCEAN
# =============================================================================

def load_wpp_survey(filepath):
    """Load WPP survey data with OCEAN personality traits."""
    print("\n" + "="*80)
    print("STEP 2: LOADING WPP SURVEY DATA")
    print("="*80)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WPP survey file not found: {filepath}")
    
    print(f"Loading WPP survey from: {filepath}")
    wpp = pd.read_excel(filepath, header=1)
    print(f"✓ Loaded {len(wpp):,} WPP respondents")
    
    return wpp


def map_wpp_demographics(wpp_df):
    """Map WPP demographics to census-compatible format."""
    df = wpp_df.copy()
    
    # Age mapping
    def map_age_group(age):
        if pd.isna(age) or age < 18:
            return None
        elif 18 <= age <= 25:
            return "18-25 (Gen Z)"
        elif 26 <= age <= 41:
            return "26-41 (Millennials)"
        elif 42 <= age <= 55:
            return "42-55 (Gen X)"
        else:
            return "56+ (Older)"
    
    df['wpp_age_group'] = df['A3r1'].apply(map_age_group)
    
    # Gender mapping
    df['wpp_gender'] = df['A2'].map({1: "Male", 2: "Female"})
    
    # Employment mapping
    employment_map = {
        1: "Working full time",
        2: "Working full time",
        3: "Working full time",
        4: "Not in labor force",
        5: "Unemployed, in between jobs",
        6: "Not in labor force",
        7: "Not in labor force"
    }
    df['wpp_employment'] = df['A5'].map(employment_map)
    
    # Income mapping (approximate Vietnamese to USD)
    income_map = {
        18: "$0 - $1,999",
        19: "$0 - $1,999",
        20: "$2,000 - $3,999",
        21: "$4,000 - $5,999",
        22: "$0 - $1,999"
    }
    df['wpp_income_range'] = df['A6'].map(income_map)
    
    # Create matching key
    df['matching_key'] = (
        df['wpp_age_group'].astype(str) + '|' +
        df['wpp_gender'].astype(str) + '|' +
        df['wpp_employment'].astype(str) + '|' +
        df['wpp_income_range'].astype(str)
    )
    
    # Filter valid records
    df = df[df['wpp_age_group'].notna()].copy()
    
    print(f"✓ Mapped {len(df):,} WPP respondents to census format")
    print(f"✓ Unique WPP profiles: {df['matching_key'].nunique():,}")
    
    return df


def extract_ocean_traits(wpp_df):
    """Extract OCEAN personality traits from WPP survey."""
    ocean_cols = {
        'openness': 'Qfeed1_subsr1',
        'conscientiousness': 'Qfeed1_subsr2',
        'extraversion': 'Qfeed1_subsr3',
        'agreeableness': 'Qfeed1_subsr4',
        'neuroticism': 'Qfeed1_subsr5'
    }
    
    ocean_df = wpp_df[['matching_key'] + list(ocean_cols.values())].copy()
    ocean_df = ocean_df.rename(columns={v: k for k, v in ocean_cols.items()})
    
    print(f"✓ Extracted OCEAN traits for {len(ocean_df):,} respondents")
    
    return ocean_df


# =============================================================================
# STEP 3: MATCH CENSUS TO WPP AND ASSIGN OCEAN
# =============================================================================

def match_and_assign_ocean(census_df, wpp_ocean_df, random_state=42):
    """Match census personas to WPP respondents and assign OCEAN traits."""
    print("\n" + "="*80)
    print("STEP 3: MATCHING CENSUS TO WPP AND ASSIGNING OCEAN")
    print("="*80)
    
    np.random.seed(random_state)
    
    # Group WPP data by matching key
    wpp_by_key = wpp_ocean_df.groupby('matching_key')
    
    # Track statistics
    exact_matches = 0
    fallback_matches = 0
    
    # Assign OCEAN traits
    ocean_traits = []
    
    print(f"Matching {len(census_df):,} census personas to WPP...")
    
    for idx, row in tqdm(census_df.iterrows(), total=len(census_df), desc="Matching WPP", ncols=100):
        matching_key = row['matching_key']
        
        # Try exact match
        if matching_key in wpp_by_key.groups:
            wpp_matches = wpp_by_key.get_group(matching_key)
            sampled = wpp_matches.sample(n=1, random_state=random_state + idx)
            exact_matches += 1
        else:
            # Fallback: random sample from all WPP
            sampled = wpp_ocean_df.sample(n=1, random_state=random_state + idx)
            fallback_matches += 1
        
        # Extract OCEAN traits
        ocean_traits.append({
            'openness': sampled['openness'].values[0],
            'conscientiousness': sampled['conscientiousness'].values[0],
            'extraversion': sampled['extraversion'].values[0],
            'agreeableness': sampled['agreeableness'].values[0],
            'neuroticism': sampled['neuroticism'].values[0]
        })
        
        # Progress update
        if (idx + 1) % 100000 == 0:
            print(f"  Progress: {idx + 1:,}/{len(census_df):,}")
    
    # Add OCEAN to census data
    traits_df = pd.DataFrame(ocean_traits)
    result_df = pd.concat([census_df.reset_index(drop=True), traits_df], axis=1)
    
    print(f"\n✓ Matching complete:")
    print(f"  Exact matches: {exact_matches:,} ({exact_matches/len(census_df)*100:.1f}%)")
    print(f"  Fallback matches: {fallback_matches:,} ({fallback_matches/len(census_df)*100:.1f}%)")
    
    return result_df


# =============================================================================
# STEP 4: SAVE OUTPUT
# =============================================================================

def save_output(df):
    """Save final dataset with OCEAN traits."""
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    # Save full dataset
    print(f"Saving full dataset to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✓ Saved {len(df):,} records")
    
    # Save preview
    print(f"Saving preview to: {OUTPUT_PREVIEW}")
    df.head(1000).to_csv(OUTPUT_PREVIEW, index=False)
    print(f"✓ Saved 1,000 records")
    
    # Print summary
    print("\n" + "="*80)
    print("WPP OCEAN ASSIGNMENT COMPLETE")
    print("="*80)
    print(f"Total personas: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nOCEAN traits (text-based 7-point scale):")
    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        unique_vals = df[trait].nunique()
        print(f"  {trait.capitalize():18s}: {unique_vals} unique levels")
    print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*24 + "WPP OCEAN ASSIGNMENT STRATEGY" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Load shared synthetic census
    print("Loading shared synthetic census population...")
    if not SYNTHETIC_CENSUS_PATH.exists():
        print(f"ERROR: Synthetic census not found: {SYNTHETIC_CENSUS_PATH}")
        print("\nPlease run first: python generate_synthetic_census.py")
        sys.exit(1)
    
    census_df = pd.read_parquet(SYNTHETIC_CENSUS_PATH)
    print(f"✓ Loaded {len(census_df):,} synthetic census records")
    
    # Step 1: Apply WPP mappings
    census_mapped = apply_wpp_mappings(census_df)
    
    # Step 2: Load WPP survey and extract OCEAN
    wpp_df = load_wpp_survey(WPP_SURVEY_PATH)
    wpp_mapped = map_wpp_demographics(wpp_df)
    wpp_ocean = extract_ocean_traits(wpp_mapped)
    
    # Step 2: Match and assign OCEAN
    final_df = match_and_assign_ocean(census_mapped, wpp_ocean, random_state=RANDOM_SEED)
    
    # Step 3: Save output
    save_output(final_df)
    
    print("\n✓ WPP strategy complete!")
    print(f"✓ Output: {OUTPUT_PARQUET}\n")


if __name__ == "__main__":
    main()
