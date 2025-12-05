"""
Synthetic Population Generation

Functions for:
1. Generating synthetic population via stratified sampling
2. Validating synthetic data quality
3. Saving synthetic population data
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats as ss


def generate_synthetic_population(df, target_n=1_000_000, seed=42):
    """
    Generate synthetic population using stratified sampling.
    
    Preserves demographic distributions while creating new synthetic records.
    
    Args:
        df: Input dataframe with WPP mappings
        target_n: Target number of synthetic records
        seed: Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Synthetic population
    """
    print(f"\nGenerating synthetic population (target: {target_n:,} records)...")
    
    # Prepare data for sampling
    d = df.copy()
    
    # Convert to appropriate types
    d['age'] = d['age'].astype(int)
    d['annual_income_usd'] = pd.to_numeric(d['annual_income_usd'], errors='coerce')
    d['weekly_work_hours'] = pd.to_numeric(d['weekly_work_hours'], errors='coerce')
    
    # Convert categorical columns
    cat_cols = ['gender', 'state', 'race', 'educational_attainment',
                'occupation', 'employment_status']
    for c in cat_cols:
        if c in d.columns:
            d[c] = d[c].fillna("Not Applicable").astype("category")
    
    # Stratified sampling by gender and age
    rng = np.random.default_rng(seed)
    
    # Create age bins for stratification
    age_bins = [18, 25, 41, 55, 91]
    d["age_bin"] = pd.cut(d["age"], bins=age_bins, right=False)
    
    # Stratify by gender and age
    strata = ["gender", "age_bin"]
    
    print(f"  Calculating stratum weights...")
    strata_weights = d.groupby(strata, observed=True).size()
    strata_weights = strata_weights / strata_weights.sum()
    
    print(f"  Sampling across {len(strata_weights)} strata...")
    samples = []
    
    for idx, proportion in strata_weights.items():
        stratum_data = d[(d["gender"] == idx[0]) & (d["age_bin"] == idx[1])]
        
        if len(stratum_data) == 0:
            continue
        
        n_samples = int(round(proportion * target_n))
        if n_samples == 0:
            continue
        
        sampled = stratum_data.sample(n=n_samples, replace=True, random_state=seed)
        samples.append(sampled)
    
    syn = pd.concat(samples, ignore_index=True)
    
    # Drop temporary age_bin column before returning
    if 'age_bin' in syn.columns:
        syn = syn.drop(columns=['age_bin'])
    
    print(f"  ✓ Generated {len(syn):,} synthetic records")
    
    return syn


def jensen_shannon_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return (ss.entropy(p, m) + ss.entropy(q, m)) / 2


def validate_synthetic_data(real_df, syn_df):
    """
    Validate synthetic data distribution matches real data.
    
    Args:
        real_df: Original real data with WPP mappings
        syn_df: Synthetic data to validate
    """
    print("\nValidating synthetic population...")
    
    validation_cols = ['wpp_age_group', 'wpp_gender', 'wpp_employment', 'wpp_income_range']
    
    print("  Jensen-Shannon divergence (lower is better):")
    for col in validation_cols:
        if col in real_df.columns and col in syn_df.columns:
            real_dist = real_df[col].value_counts().sort_index()
            syn_dist = syn_df[col].value_counts().sort_index()
            syn_dist = syn_dist.reindex(real_dist.index, fill_value=0)
            
            js_score = jensen_shannon_divergence(real_dist.values, syn_dist.values)
            print(f"    JS({col:25s}) = {js_score:.6f}")
    
    print("  ✓ Validation complete")


def save_wpp_ready_data(df, parquet_file="census_wpp_ready_1m.parquet", 
                        csv_file="census_wpp_ready_preview.csv"):
    """
    Save WPP-ready synthetic data.
    
    Args:
        df: WPP-ready synthetic population
        parquet_file: Output parquet file path
        csv_file: Output CSV preview file path
    
    Returns:
        pd.DataFrame: Final WPP-ready data
    """
    print("\nSaving WPP-ready data...")
    
    # Select final columns
    final_output_cols = [
        'age', 'gender', 'state', 'race',
        'educational_attainment', 'occupation',
        'employment_status', 'annual_income_usd', 'weekly_work_hours',
        'wpp_age_group', 'wpp_gender', 'wpp_employment', 'wpp_income_range',
        'matching_key'
    ]
    
    # Only select columns that exist
    available_output_cols = [c for c in final_output_cols if c in df.columns]
    wpp_ready_final = df[available_output_cols].copy()
    
    # Save to parquet
    wpp_ready_final.to_parquet(parquet_file, index=False)
    print(f"  ✓ Saved full data to: {parquet_file}")
    print(f"    Size: {len(wpp_ready_final):,} records")
    
    # Save preview to CSV
    wpp_ready_final.head(1000).to_csv(csv_file, index=False)
    print(f"  ✓ Saved preview to: {csv_file}")
    print(f"    Size: 1,000 records")
    
    return wpp_ready_final


def print_summary(df):
    """Print summary statistics for WPP-ready data."""
    print("\n" + "="*80)
    print("✓ SYNTHETIC WPP-READY POPULATION GENERATED")
    print("="*80)
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique matching profiles: {df['matching_key'].nunique():,}")
    
    print(f"\n📋 WPP Matching Field Distributions:")
    for field in ['wpp_age_group', 'wpp_gender', 'wpp_employment', 'wpp_income_range']:
        if field in df.columns:
            print(f"\n  {field}:")
            dist = df[field].value_counts().sort_index()
            for val, count in dist.items():
                pct = count / len(df) * 100
                print(f"    {val:30s}: {count:7,} ({pct:5.2f}%)")
    
    print(f"\n{'='*80}")
    print("✅ READY FOR WPP PERSONALITY MATCHING!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Load WPP survey data (wpp_merged_result_VT.xlsx)")
    print("  2. Match Census personas to WPP respondents using 'matching_key'")
    print("  3. Sample OCEAN traits from matched WPP respondents")
    print("  4. Create final synthetic population with demographics + personalities")


def process_synthesis(mapped_data, target_n=1_000_000, seed=42,
                     parquet_output="census_wpp_ready_1m.parquet",
                     csv_output="census_wpp_ready_preview.csv"):
    """
    Main function for synthetic population generation.
    
    Args:
        mapped_data: WPP-mapped census data
        target_n: Target number of synthetic records
        seed: Random seed for reproducibility
        parquet_output: Output parquet file path
        csv_output: Output CSV preview file path
    
    Returns:
        pd.DataFrame: WPP-ready synthetic population
    """
    print("\n" + "="*80)
    print("STEP 3: SYNTHETIC POPULATION GENERATION")
    print("="*80)
    
    # Generate synthetic population
    df_synthetic = generate_synthetic_population(mapped_data, target_n=target_n, seed=seed)
    
    # Validate synthetic data
    validate_synthetic_data(mapped_data, df_synthetic)
    
    # Save WPP-ready data
    wpp_ready = save_wpp_ready_data(df_synthetic, parquet_output, csv_output)
    
    # Print summary
    print_summary(wpp_ready)
    
    return wpp_ready
