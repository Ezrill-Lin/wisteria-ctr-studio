"""
Utility functions for v2 persona generation pipeline

This module contains helper functions for:
- Loading data and templates
- Formatting demographics and OCEAN scores
- Creating prompts
- Saving output files
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def load_system_message():
    """Load the system message for v2 persona generation"""
    system_file = Path("prompts") / "system_message.txt"
    
    if not system_file.exists():
        raise FileNotFoundError(f"System message file not found: {system_file}")
    
    with open(system_file, 'r', encoding='utf-8') as f:
        system_message = f.read()
    
    print(f"✓ Loaded system message: {system_file}")
    return system_message


def load_prompt_template(strategy):
    """Load the persona generation prompt template for a strategy"""
    prompt_file = Path("prompts") / f"{strategy}_persona_gen_prompt.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read()
    
    print(f"✓ Loaded prompt template: {prompt_file}")
    return template


def load_ocean_data(strategy, sample_size=None):
    """
    Load census data with OCEAN scores for a given strategy
    
    Args:
        strategy: Strategy name (random/ipip/wpp)
        sample_size: Number of records to load (None = all)
    
    Returns:
        DataFrame with demographics and OCEAN scores
    """
    
    # Data files in personas_v2 folder
    file_mapping = {
        'random': 'random_matching/census_random_ocean_1m.parquet',
        'ipip': 'ipip_matching/census_ipip_ocean_1m.parquet',
        'wpp': 'wpp_matching/census_with_personality_1m.parquet'
    }
    
    if strategy not in file_mapping:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(file_mapping.keys())}")
    
    data_file = Path(file_mapping[strategy])
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Define columns needed
    if strategy == 'wpp':
        ocean_cols = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    else:
        ocean_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    demographic_cols = ['age', 'gender', 'occupation', 'educational_attainment', 'state', 'employment_status']
    columns_to_read = demographic_cols + ocean_cols
    
    # Sample if requested
    if sample_size:
        parquet_file = pq.ParquetFile(data_file)
        total_rows = parquet_file.metadata.num_rows
        
        row_indices = sorted(np.random.choice(total_rows, min(sample_size, total_rows), replace=False))
        
        df = pd.read_parquet(data_file, columns=columns_to_read)
        df = df.iloc[row_indices]
        
        print(f"✓ Sampled {len(df):,} records from {data_file} (total: {total_rows:,})")
    else:
        df = pd.read_parquet(data_file, columns=columns_to_read)
        print(f"✓ Loaded {len(df):,} records from {data_file}")
    
    return df


def format_demographics(row):
    """Format demographic information into readable text"""
    demo_parts = []
    
    if 'age' in row and pd.notna(row['age']):
        demo_parts.append(f"{int(row['age'])}-year-old")
    if 'gender' in row and pd.notna(row['gender']):
        demo_parts.append(str(row['gender']).lower())
    
    if 'occupation' in row and pd.notna(row['occupation']):
        demo_parts.append(f"working as {row['occupation']}")
    
    if 'educational_attainment' in row and pd.notna(row['educational_attainment']):
        demo_parts.append(f"with {row['educational_attainment']}")
    
    if 'state' in row and pd.notna(row['state']):
        demo_parts.append(f"living in {row['state']}")
    
    if 'employment_status' in row and pd.notna(row['employment_status']):
        demo_parts.append(f"({row['employment_status']})")
    
    return ", ".join(demo_parts)


def format_ocean_scores(row, strategy):
    """Format OCEAN scores based on strategy type"""
    
    if strategy == 'random' or strategy == 'ipip':
        ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        scores = {trait: round(row[trait], 2) for trait in ocean_traits if trait in row}
    
    elif strategy == 'wpp':
        ocean_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        scores = {trait.title(): row[trait] for trait in ocean_traits if trait in row}
    
    return scores


def create_persona_prompt(template, demographics, ocean_scores, strategy, persona_id):
    """
    Create the full prompt for v2 LLM persona generation
    
    Args:
        template: Prompt template with trait guidance and output format
        demographics: Formatted demographic string
        ocean_scores: Dictionary of OCEAN trait scores
        strategy: Strategy name (random/ipip/wpp)
        persona_id: ID to include in the output
    
    Returns:
        Full prompt string ready for LLM
    """
    
    # Build the data section
    data_section = "\n### Input Data\n\n"
    data_section += f"**Persona ID:** {persona_id}\n\n"
    data_section += f"**Demographics:** {demographics}\n\n"
    data_section += "**OCEAN Scores:**\n"
    
    for trait, score in ocean_scores.items():
        if strategy == 'wpp':
            data_section += f"- {trait}: {score}\n"
        else:
            data_section += f"- {trait}: {score}/10\n"
    
    # Combine template with data
    full_prompt = template + "\n" + data_section
    
    return full_prompt


def save_personas(personas, strategy, output_dir=".", append=True):
    """
    Save generated v2 personas to JSONL file
    
    Args:
        personas: List of persona dictionaries
        strategy: Strategy name (random/ipip/wpp)
        output_dir: Base output directory
        append: If True, append to existing file; if False, overwrite
    
    Returns:
        Path to saved JSONL file
    """
    
    output_dir = Path(output_dir) / f"{strategy}_matching"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_file = output_dir / f"personas_{strategy}_v2.jsonl"
    
    # Count existing records
    existing_count = 0
    if append:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for _ in f)
            print(f"  Found {existing_count:,} existing personas")
        except Exception as e:
            print(f"  Creating new file: {jsonl_file}")
    else:
        if jsonl_file.exists():
            print(f"  Overwriting existing file: {jsonl_file}")
        else:
            print(f"  Creating new file: {jsonl_file}")
    
    # Write to file
    mode = 'a' if append else 'w'
    with open(jsonl_file, mode, encoding='utf-8') as f:
        for persona in personas:
            f.write(json.dumps(persona, ensure_ascii=False) + '\n')
    
    total_count = (existing_count if append else 0) + len(personas)
    print(f"✓ Saved {total_count:,} total personas to {jsonl_file}")
    if append and existing_count > 0:
        print(f"  ({existing_count:,} existing + {len(personas):,} new)")
    
    return jsonl_file


def clean_failed_personas(strategy, output_dir="."):
    """
    Remove entries with '[FAILED]' in persona_description from JSONL file
    
    Args:
        strategy: Strategy name (random/ipip/wpp)
        output_dir: Base output directory
    
    Returns:
        Tuple of (cleaned_count, removed_count)
    """
    
    output_dir = Path(output_dir) / f"{strategy}_matching"
    jsonl_file = output_dir / f"personas_{strategy}_v2.jsonl"
    
    if not jsonl_file.exists():
        print(f"  No file to clean: {jsonl_file}")
        return 0, 0
    
    # Read all personas
    valid_personas = []
    failed_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                persona = json.loads(line.strip())
                # Check if persona_description contains '[FAILED]'
                if '[FAILED]' in persona.get('persona_description', ''):
                    failed_count += 1
                else:
                    valid_personas.append(persona)
            except json.JSONDecodeError:
                continue
    
    if failed_count == 0:
        print(f"✓ No failed personas found in {jsonl_file}")
        return len(valid_personas), 0
    
    # Overwrite with valid personas only
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for persona in valid_personas:
            f.write(json.dumps(persona, ensure_ascii=False) + '\n')
    
    print(f"✓ Cleaned {jsonl_file}")
    print(f"  Removed {failed_count:,} failed persona(s), {len(valid_personas):,} valid persona(s) remaining")
    
    return len(valid_personas), failed_count
