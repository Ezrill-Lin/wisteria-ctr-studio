"""
Synthetic Population Agent

This script uses generated personas to simulate human responses to tasks.
It loads personas (with demographics + OCEAN + descriptions) and has them 
respond to questions via LLM, simulating human-like behavior.

Input:
    - personas/{strategy}_matching/personas_{strategy}.jsonl: Generated personas
    - questions.json: Questions to ask personas
    
Output:
    - responses_{strategy}.jsonl: Synthetic responses

Usage:
    python agent.py --strategy random --model gpt-4o-mini --response-version v2 --sample-size 2000 --concurrent 10
    python agent.py --strategy wpp --model gpt-4o-mini --response-version v2 --sample-size 2000 --concurrent 10
    python agent.py --no-persona --model gpt-4o-mini --response-version v2 --sample-size 2000 --concurrent 10

    python agent.py --strategy random --model gpt-4o-mini --response-version v1 --sample-size 2000 --concurrent 10
    python agent.py --strategy wpp --model gpt-4o-mini --response-version v1 --sample-size 2000 --concurrent 10
    python agent.py --no-persona --model gpt-4o-mini --response-version v1 --sample-size 2000 --concurrent 10
"""

import argparse
import asyncio

from agent_utils import save_responses_batch, create_prompt, load_personas, load_questions, analyze_responses, clean_failed_responses, clean_duplicated_responses, detect_api_provider
from client import collect_responses


def main(strategy='random', sample_size=100, questions_file=None, 
         model="gpt-4o-mini", concurrent_requests=20, api_provider="openai", no_persona=False, no_clean_duplicated=False, response_version='v1'):
    """Main synthetic agent pipeline
    
    Args:
        strategy: Persona strategy to use (random/ipip/wpp)
        sample_size: Number of personas to use
        questions_file: Path to questions JSON
        model: Model to use
        concurrent_requests: Number of concurrent API calls
        api_provider: API provider ('openai' or 'deepseek')
        no_persona: If True, generate responses without persona context
        no_clean_duplicated: If True, skip cleaning duplicated persona_id responses
        response_version: Response version folder ('v1' or 'v2')
    """
    print("\n" + "="*80)
    print("SYNTHETIC AGENT PIPELINE")
    print("="*80)
    print(f"Mode: {'NO PERSONA (baseline)' if no_persona else 'WITH PERSONA'}")
    if not no_persona:
        print(f"Strategy: {strategy}")
    print(f"Sample size: {sample_size:,}")
    print(f"Model: {model}")
    print(f"Concurrent requests: {concurrent_requests}")
    print("="*80 + "\n")
    
    # Use default questions file if not provided
    if questions_file is None:
        questions_file = "test/test_questions.json"
    
    # Load personas and questions
    if no_persona:
        # Create dummy personas (just IDs) for no-persona mode
        personas = [{'id': i} for i in range(1, sample_size + 1)]
        print(f"✓ Created {len(personas)} baseline responses (no persona)")
        strategy = 'no_persona'  # Override strategy for output file naming
    else:
        personas = load_personas(strategy, sample_size=sample_size, version=response_version)
    questions = load_questions(questions_file)
    
    # Collect responses
    responses_file = asyncio.run(collect_responses(
        personas, questions, strategy, 
        model=model, 
        api_provider=api_provider,
        concurrent_requests=concurrent_requests,
        response_version=response_version
    ))
    
    # Analyze responses
    analyze_responses(responses_file)
    
    # Clean failed responses
    print("\nCleaning failed responses...")
    clean_failed_responses(responses_file)
    
    # Clean duplicated responses
    if not no_clean_duplicated:
        print("\nCleaning duplicated responses...")
        clean_duplicated_responses(responses_file)
    else:
        print("\nSkipping duplicate cleaning (--no-clean-duplicated flag set)")
    
    print("\n" + "="*80)
    print("✅ SYNTHETIC AGENT COMPLETE")
    print("="*80)
    print(f"\nResponses saved to: {responses_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic responses using personas")
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'ipip', 'wpp'],
                       help='Persona strategy to use')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of personas to use')
    parser.add_argument('--questions', type=str, default=None,
                       help='Path to questions JSON file')
    parser.add_argument('--model', type=str, default='deepseek-chat',
                       help='Model to use (default: gpt-4o-mini for OpenAI, deepseek-chat for DeepSeek, gemini-2.5-flash for Gemini)')
    parser.add_argument('--concurrent', type=int, default=20,
                       help='Number of concurrent API requests')
    parser.add_argument('--no-persona', action='store_true',
                       help='Generate responses without persona context (baseline comparison)')
    parser.add_argument('--no-clean-duplicated', action='store_true',
                       help='Skip cleaning duplicated persona_id responses')
    parser.add_argument('--response-version', type=str, default='v2',
                       choices=['v1', 'v2'],
                       help='Response version folder (default: v1)')

    args = parser.parse_args()

    # Auto-detect API provider from model name
    api_provider = detect_api_provider(args.model)

    main(
        strategy=args.strategy,
        sample_size=args.sample_size,
        questions_file=args.questions,
        model=args.model,
        concurrent_requests=args.concurrent,
        api_provider=api_provider,
        no_persona=args.no_persona,
        no_clean_duplicated=args.no_clean_duplicated,
        response_version=args.response_version
    )
