"""
Test Prompt Generation

This script outputs the actual prompt that will be sent to the LLM for persona generation.
Useful for debugging and understanding what the model sees.

Usage:
    python test_prompt.py --strategy random
    python test_prompt.py --strategy ipip --row-index 5
    python test_prompt.py --strategy wpp --output prompt.txt
"""

import argparse
from pathlib import Path

from utils import (
    load_prompt_template,
    load_ocean_data,
    format_demographics,
    format_ocean_scores,
    create_persona_prompt
)


def main(strategy='random', row_index=0, output_file=None):
    """
    Generate and display/save the prompt for a sample persona
    
    Args:
        strategy: Which OCEAN assignment strategy (random/ipip/wpp)
        row_index: Which row from the data to use (default: 0)
        output_file: If provided, save to this file instead of printing
    """
    
    print("\n" + "="*80)
    print("PROMPT GENERATION TEST")
    print("="*80)
    print(f"Strategy: {strategy}")
    print(f"Row index: {row_index}")
    print("="*80 + "\n")
    
    # Load prompt template
    print("Loading prompt template...")
    template = load_prompt_template(strategy)
    print(f"✓ Template loaded ({len(template)} characters)\n")
    
    # Load sample data (just 10 rows for efficiency)
    print("Loading sample OCEAN data...")
    df = load_ocean_data(strategy, sample_size=10)
    
    if row_index >= len(df):
        print(f"⚠️  Warning: Row index {row_index} out of range (max: {len(df)-1}), using row 0")
        row_index = 0
    
    row = df.iloc[row_index]
    print(f"✓ Using row {row_index}\n")
    
    # Format demographics
    demographics = format_demographics(row)
    print("Demographics:")
    print(f"  {demographics}\n")
    
    # Format OCEAN scores
    ocean_scores = format_ocean_scores(row, strategy)
    print("OCEAN Scores:")
    for trait, score in ocean_scores.items():
        if strategy == 'wpp':
            print(f"  {trait}: {score}")
        else:
            print(f"  {trait}: {score}/10")
    print()
    
    # Create full prompt
    full_prompt = create_persona_prompt(template, demographics, ocean_scores, strategy)
    
    print("="*80)
    print("FULL PROMPT")
    print("="*80)
    
    # Output to file or console
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        print(f"\n✓ Prompt saved to: {output_path}")
        print(f"  Length: {len(full_prompt)} characters")
        print(f"  Lines: {full_prompt.count(chr(10)) + 1}")
    else:
        print(full_prompt)
        print("\n" + "="*80)
        print(f"Prompt length: {len(full_prompt)} characters")
        print(f"Prompt lines: {full_prompt.count(chr(10)) + 1}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and view the persona generation prompt")
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'ipip', 'wpp'],
                       help='OCEAN assignment strategy to use')
    parser.add_argument('--row-index', type=int, default=0,
                       help='Which row from the data to use (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save prompt to this file instead of printing to console')
    
    args = parser.parse_args()
    
    main(strategy=args.strategy, row_index=args.row_index, output_file=args.output)
