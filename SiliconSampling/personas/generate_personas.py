"""
Usage:
    # Generate 100 personas using random strategy (default)
    python generate_personas.py --strategy random --sample-size 100
    
    # Overwrite existing file instead of appending
    python generate_personas.py --strategy random --sample-size 200 --no-append
    
    # Skip automatic cleaning of failed personas
    python generate_personas.py --strategy ipip --sample-size 1000 --no-cleaning
"""

import os
import json
import time
import argparse
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
import numpy as np
import pyarrow.parquet as pq

from utils import (
    load_prompt_template,
    load_ocean_data,
    format_demographics,
    format_ocean_scores,
    create_persona_prompt,
    save_personas,
    clean_failed_personas
)


async def call_openai_api_async(client, prompt, model="gpt-4o-mini", max_retries=3):
    """
    Async call to OpenAI API to generate persona description
    
    Args:
        client: AsyncOpenAI client instance
        prompt: Full prompt text
        model: OpenAI model to use
        max_retries: Number of retry attempts for failed calls
    
    Returns:
        Generated persona description (str) or None if failed
    """
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a psychological profiling expert. Generate vivid, coherent persona descriptions that accurately reflect provided demographic and personality data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result.get('persona_description', '')
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
    
    return None


async def generate_personas_batch(df, template, strategy, model="gpt-4o-mini", batch_save_size=100, concurrent_requests=50, append=True, auto_clean=True):
    """
    Generate persona descriptions using async batch processing for speed
    Saves progress every batch_save_size personas to prevent data loss
    
    Args:
        df: DataFrame with demographics and OCEAN scores (already sampled)
        template: Prompt template
        strategy: Strategy name
        model: OpenAI model to use
        batch_save_size: Save to file every N personas (default: 100)
        concurrent_requests: Number of concurrent API calls (default: 50)
        append: If True (default), append to existing file; if False, overwrite
        auto_clean: If True (default), automatically clean failed personas after generation
    
    Returns:
        Total count of personas generated
    """
    
    # Check API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables!\n"
            "Set it with: $env:OPENAI_API_KEY='your-key-here' (PowerShell)"
        )
    
    # Create async client
    client = AsyncOpenAI(api_key=api_key)
    
    # Prepare all prompts first (use original DataFrame index for tracking back to source)
    tasks_data = []
    for idx, row in df.iterrows():
        demographics = format_demographics(row)
        ocean_scores = format_ocean_scores(row, strategy)
        prompt = create_persona_prompt(template, demographics, ocean_scores, strategy)
        tasks_data.append((int(idx), demographics, ocean_scores, prompt))
    
    # Process in concurrent batches
    all_personas = []
    failed_count = 0
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    async def process_one(idx, demographics, ocean_scores, prompt):
        async with semaphore:
            persona_description = await call_openai_api_async(client, prompt, model=model)
            if persona_description is None:
                nonlocal failed_count
                failed_count += 1
                persona_description = f"[FAILED] API call failed after retries"
            
            return {
                'id': idx,
                'demographics': demographics,
                'ocean_scores': ocean_scores,
                'persona_description': persona_description
            }
    
    # Create all tasks
    tasks = [process_one(idx, demo, scores, prompt) for idx, demo, scores, prompt in tasks_data]
    
    # Run with progress bar
    print("\nProcessing API calls...")
    results = []
    first_batch = True
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Generating personas", ncols=100):
        result = await coro
        results.append(result)
        
        # Save batch when reaching batch_save_size
        if len(results) >= batch_save_size:
            # First batch uses the append parameter, subsequent batches always append
            save_personas(results, strategy, append=(append if first_batch else True))
            first_batch = False
            results = []  # Clear batch
    
    # Save any remaining personas
    if results:
        save_personas(results, strategy, append=(append if first_batch else True))
    
    if failed_count > 0:
        print(f"\n⚠ Warning: {failed_count}/{len(tasks_data)} persona generations failed")
    
    # Auto-clean failed personas if enabled
    if auto_clean:
        print("\nCleaning failed personas...")
        valid_count, removed_count = clean_failed_personas(strategy)
    
    return len(tasks_data)


def main(strategy='random', sample_size=100, model="gpt-4o-mini", concurrent_requests=20, append=True, auto_clean=True):
    """
    Main persona generation pipeline
    
    Args:
        strategy: Which OCEAN assignment strategy to use (random/ipip/wpp)
        sample_size: Number of personas to generate (for testing)
        model: OpenAI model to use
        concurrent_requests: Number of concurrent API calls
        append: If True (default), append to existing file; if False, overwrite
    """
    
    print("\n" + "="*80)
    print("PERSONA GENERATION PIPELINE")
    print("="*80)
    print(f"Strategy: {strategy}")
    print(f"Sample size: {sample_size:,} (use None for full dataset)")
    print(f"Model: {model}")
    print(f"Concurrent requests: {concurrent_requests}")
    print("="*80 + "\n")
    
    # Load prompt template
    template = load_prompt_template(strategy)
    
    # Load OCEAN data (optimized with sampling during read)
    df = load_ocean_data(strategy, sample_size=sample_size)
    
    # Generate personas (async batch processing)
    total_count = asyncio.run(generate_personas_batch(df, template, strategy, model, concurrent_requests=concurrent_requests, append=append, auto_clean=auto_clean))
    
    # Get output file path
    from pathlib import Path
    output_file = Path(f"{strategy}_matching") / f"personas_{strategy}.jsonl"
    
    print("\n" + "="*80)
    print("✅ PERSONA GENERATION COMPLETE")
    print("="*80)
    print(f"\nTotal personas generated: {total_count:,}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate persona descriptions from OCEAN scores using OpenAI API")
    parser.add_argument('--strategy', type=str, default='random', 
                       choices=['random', 'ipip', 'wpp'],
                       help='OCEAN assignment strategy to use')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of personas to generate (for testing)')
    parser.add_argument('--full', action='store_true',
                       help='Generate personas for full dataset')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--concurrent', type=int, default=20,
                       help='Number of concurrent API requests (default: 20)')
    parser.add_argument('--no-append', action='store_true',
                       help='Overwrite existing personas file instead of appending')
    parser.add_argument('--no-cleaning', action='store_true',
                       help='Skip automatic cleaning of failed personas after generation')
    
    args = parser.parse_args()
    
    sample_size = None if args.full else args.sample_size
    
    main(strategy=args.strategy, sample_size=sample_size, model=args.model, concurrent_requests=args.concurrent, append=not args.no_append, auto_clean=not args.no_cleaning)
