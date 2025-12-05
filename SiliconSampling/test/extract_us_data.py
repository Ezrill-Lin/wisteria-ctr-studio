"""
Extract US Ground Truth Data

This script filters the big5data.csv to extract only US respondents
and keeps only the relevant IPIP-50 item columns (E1-E10, N1-N10, A1-A10, C1-C10, O1-O10).

Input:
    - test/big5data.csv: Full ground truth dataset
    
Output:
    - test/us_ground_truth.csv: US-only data with IPIP-50 items

Usage:
    python test/extract_us_data.py
"""

import pandas as pd
from pathlib import Path


def extract_us_ground_truth(input_file="big5data.csv", output_file="us_ground_truth.csv"):
    """
    Extract US respondents and relevant IPIP-50 columns
    
    Args:
        input_file: Path to full ground truth data
        output_file: Path to save US-only data
    """
    print(f"\n{'='*80}")
    print("EXTRACTING US GROUND TRUTH DATA")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"✓ Loaded {len(df):,} total records")
    print(f"  Columns: {list(df.columns)}")
    
    # Filter for US only
    print(f"\nFiltering for country='US'...")
    df_us = df[df['country'] == 'US'].copy()
    print(f"✓ Found {len(df_us):,} US respondents ({len(df_us)/len(df)*100:.1f}% of total)")
    
    # Extract IPIP-50 item columns
    ipip_cols = []
    for trait in ['E', 'N', 'A', 'C', 'O']:
        for i in range(1, 11):
            col = f"{trait}{i}"
            if col in df_us.columns:
                ipip_cols.append(col)
    
    print(f"\nExtracting {len(ipip_cols)} IPIP-50 item columns...")
    df_us_ipip = df_us[ipip_cols].copy()
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_us_ipip.to_csv(output_path, index=False)
    
    print(f"✓ Saved to {output_file}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total records: {len(df):,}")
    print(f"US records: {len(df_us):,} ({len(df_us)/len(df)*100:.1f}%)")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")
    
    # Show sample statistics
    print("Sample statistics (first 5 items):")
    for col in ipip_cols[:5]:
        print(f"  {col}: mean={df_us_ipip[col].mean():.2f}, std={df_us_ipip[col].std():.2f}")
    
    return df_us_ipip


if __name__ == "__main__":
    extract_us_ground_truth()
