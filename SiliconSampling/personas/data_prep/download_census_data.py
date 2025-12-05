"""
Census Data Download Script

This script downloads census data from the US Census API and saves it to a parquet file.
Run this script once to download the data, then use data.ipynb for processing.

Usage:
    python download_census_data.py
"""

import requests
import pandas as pd
import os


def download_census_data(api_key, output_file="acs_pums_2023.parquet"):
    """
    Download census data from the API and save to parquet file.
    
    Args:
        api_key: Census API key
        output_file: Path to save the parquet file
    
    Returns:
        pd.DataFrame: The downloaded census data
    """
    url = (
        "https://api.census.gov/data/2023/acs/acs1/pums?"
        "get=AGEP,SEX,SCHL,OCCP,INDP,PINCP,RAC3P,WKHP,ESR,STATE"
        f"&for=state:*&key={api_key}"
    )
    
    print("="*80)
    print("DOWNLOADING CENSUS DATA FROM API")
    print("="*80)
    print(f"\nFetching census data from API...")
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    
    if response.status_code != 200:
        print(f"Error: {response.text[:500]}")
        raise SystemExit("Failed to download census data")
    
    data = response.json()
    
    # Convert JSON → DataFrame
    columns = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=columns)
    
    # Save to parquet
    df.to_parquet(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("✓ DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Saved {len(df):,} records to: {output_file}")
    print(f"✓ File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"\nYou can now run data.ipynb to process this data.")
    
    return df


if __name__ == "__main__":
    # Configuration
    API_KEY = "71cbe35ff6e0c0df92b72d8a233c40a85de5b0b9"
    OUTPUT_FILE = "acs_pums_2023.parquet"
    
    # Check if file already exists
    if os.path.exists(OUTPUT_FILE):
        print("="*80)
        print("WARNING: Data file already exists!")
        print("="*80)
        print(f"File: {OUTPUT_FILE}")
        print(f"Size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
        
        response = input("\nDo you want to re-download? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nSkipping download. Using existing file.")
            print("Run data.ipynb to process the data.")
            exit(0)
        else:
            print("\nRe-downloading data...")
    
    # Download the data
    download_census_data(API_KEY, OUTPUT_FILE)
