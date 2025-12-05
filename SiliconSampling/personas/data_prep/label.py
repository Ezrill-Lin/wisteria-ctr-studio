"""
Census Data Labeling and Privacy Preservation

Functions for:
1. Loading raw census data from parquet file
2. Decoding census codes to human-readable labels
3. Applying privacy noise (age jitter, categorical shuffling)
4. Filtering and cleaning data
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json


def load_raw_data(filepath="data_prep/acs_pums_2023.parquet"):
    """Load raw census data from parquet file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            f"Please run 'python download_census_data.py' first."
        )
    
    print(f"Loading raw data from {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def load_code_reference(filepath="data_prep/code_reference.json"):
    """Load census code reference for labeling."""
    with open(filepath, 'r', encoding='utf-8') as f:
        code_ref = pd.read_json(f)
    print(f"✓ Loaded code reference from {filepath}")
    return code_ref


def build_value_map(col_spec):
    """Extract the mapping dictionary from the code reference structure."""
    if 'values' in col_spec and 'item' in col_spec['values']:
        return col_spec['values']['item']
    return {}


def map_series_with_codes(series, value_map):
    """Map a pandas Series using a value map dictionary."""
    return series.astype(str).map(value_map).fillna(series.astype(str))


def decode_census_codes(df, code_ref):
    """Decode categorical census codes to human-readable labels."""
    print("\nDecoding census codes...")
    
    # Columns to decode
    cols_to_decode = ["SEX", "SCHL", "OCCP", "INDP", "RAC3P", "ESR", "STATE"]
    cols_to_remove = []
    
    for col_code in cols_to_decode:
        if col_code in df.columns and col_code in code_ref:
            vmap = build_value_map(code_ref[col_code])
            new_col_label = code_ref[col_code].get('label', col_code)
            df[new_col_label] = map_series_with_codes(df[col_code], vmap)
            cols_to_remove.append(col_code)
            print(f"  ✓ Decoded {col_code} → {new_col_label}")
    
    # Convert numeric columns
    df['age'] = pd.to_numeric(df['AGEP'], errors='coerce')
    df['annual_income_usd'] = pd.to_numeric(df['PINCP'], errors='coerce')
    df['weekly_work_hours'] = pd.to_numeric(df['WKHP'], errors='coerce')
    cols_to_remove.extend(['AGEP', 'PINCP', 'WKHP'])
    
    # Drop original coded columns and lowercase 'state' if exists
    cols_to_remove_final = [c for c in cols_to_remove if c in df.columns]
    if 'state' in df.columns:
        cols_to_remove_final.append('state')
    df = df.drop(columns=cols_to_remove_final)
    
    # Rename labeled columns to lowercase for consistency
    column_rename_map = {
        'Sex': 'gender',
        'Educational attainment': 'educational_attainment',
        'Occupation': 'occupation',
        'Industry': 'industry',
        'Race': 'race',
        'Employment status recode': 'employment_status',
        'State': 'state'
    }
    df = df.rename(columns={k: v for k, v in column_rename_map.items() if k in df.columns})
    
    print(f"✓ Decoded data shape: {df.shape}")
    return df


def apply_privacy_noise(df, seed=42):
    """
    Apply privacy-preserving noise to the data.
    
    Privacy measures:
    - Age jitter: Add random noise ±2 years
    - Categorical shuffling: Shuffle non-identifying categorical columns
    """
    print("\nApplying privacy noise...")
    df = df.copy()
    rng = np.random.default_rng(seed)
    
    # Age jitter (±2 years)
    if 'age' in df.columns:
        original_age = df['age'].copy()
        df['age'] = (
            df['age'].astype(float) + 
            rng.integers(-2, 3, size=len(df))
        )
        df['age'] = df['age'].clip(0, 120).round().astype(int)
        print(f"  ✓ Applied age jitter (±2 years)")
    
    # Shuffle categorical columns for privacy
    cols_to_shuffle = ['state', 'race', 'educational_attainment', 'occupation']
    
    for col in cols_to_shuffle:
        if col in df.columns:
            # Reset index to ensure proper shuffling
            df = df.reset_index(drop=True)
            # Shuffle the column
            shuffled = df[col].sample(frac=1, random_state=rng.integers(0, 100000)).reset_index(drop=True)
            df[col] = shuffled.values
            print(f"  ✓ Shuffled {col}")
    
    return df


def filter_and_clean(df):
    """Filter data and remove illogical records."""
    print("\nFiltering and cleaning data...")
    
    # Filter to age 18+ (adult population)
    original_count = len(df)
    df = df[df['age'] >= 18].copy()
    print(f"  ✓ Filtered to age 18+: {len(df):,} records (removed {original_count - len(df):,})")
    
    # Remove illogical records (unemployed/not in labor force with work hours)
    invalid_mask = (
        ((df['employment_status'] == 'Unemployed') & (df['weekly_work_hours'] > 0)) |
        ((df['employment_status'] == 'Not in labor force') & (df['weekly_work_hours'] > 0))
    )
    invalid_count = invalid_mask.sum()
    df = df[~invalid_mask].copy()
    print(f"  ✓ Removed {invalid_count:,} illogical records ({100*invalid_count/(len(df)+invalid_count):.2f}%)")
    
    return df


def save_labeled_data(df, output_file="data_prep/acs_pums_2023_labeled.parquet"):
    """Save labeled data to parquet file."""
    df.to_parquet(output_file, index=False)
    print(f"\n✓ Saved labeled data to: {output_file}")
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return output_file


def process_labeling(raw_data_file="data_prep/acs_pums_2023.parquet", 
                     code_ref_file="data_prep/code_reference.json",
                     output_file="data_prep/acs_pums_2023_labeled.parquet",
                     seed=42):
    """
    Main function for census data labeling and privacy preservation.
    
    Returns:
        pd.DataFrame: Labeled and privacy-preserved data
    """
    print("="*80)
    print("STEP 1: CENSUS DATA LABELING & PRIVACY PRESERVATION")
    print("="*80)
    
    # Load data
    df = load_raw_data(raw_data_file)
    code_ref = load_code_reference(code_ref_file)
    
    # Decode census codes
    df = decode_census_codes(df, code_ref)
    
    # Apply privacy noise
    df = apply_privacy_noise(df, seed=seed)
    
    # Filter and clean
    df = filter_and_clean(df)
    
    # Save labeled data
    save_labeled_data(df, output_file)
    
    print("\n✓ Labeling complete")
    
    return df


if __name__ == "__main__":
    # Run the complete labeling pipeline
    process_labeling()
