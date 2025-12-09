"""
Test script to check the optimized prompt format for v1 and v2 personas
"""

import json
import argparse
from agent_utils import load_personas, load_questions, create_prompt


def test_prompt(version='v1', strategy='random', num_questions=5):
    """Test the prompt generation with a sample persona
    
    Args:
        version: Persona version ('v1' or 'v2')
        strategy: Strategy name ('random', 'ipip', 'wpp')
        num_questions: Number of questions to include in test
    """
    
    # Load one persona
    personas = load_personas(strategy, sample_size=1, version=version)
    persona = personas[0]
    
    # Load questions
    questions = load_questions('test/test_questions.json')
    
    # Create prompt
    system_message, user_prompt = create_prompt(persona, questions[:num_questions])
    
    # Print results
    print("="*80)
    print(f"PROMPT TEST - {version.upper()} PERSONA ({strategy.upper()} STRATEGY)")
    print("="*80)
    
    print("\n📋 PERSONA INFO:")
    print(f"ID: {persona.get('id', 'N/A')}")
    print(f"Demographics: {persona['demographics']}")
    print(f"OCEAN: {persona['ocean_scores']}")
    
    if version == 'v2':
        print("\n🎯 V2 ADDITIONAL FIELDS:")
        if 'behavioral_tendencies' in persona:
            print("Behavioral Tendencies:")
            for key, value in persona['behavioral_tendencies'].items():
                print(f"  - {key}: {value}")
        if 'self_schema' in persona:
            print(f"Self-Schema: {len(persona['self_schema'])} beliefs")
    
    print("\n📝 SYSTEM MESSAGE:")
    print("-"*80)
    print(system_message)
    print("-"*80)
    
    print("\n📝 USER PROMPT:")
    print("-"*80)
    print(user_prompt)
    print("-"*80)
    
    print("\n📊 PROMPT STATISTICS:")
    system_length = len(system_message)
    user_length = len(user_prompt)
    total_length = system_length + user_length
    estimated_tokens = total_length // 4  # Rough estimate: 1 token ≈ 4 characters
    
    print(f"System message: {system_length:,} chars (~{system_length // 4:,} tokens)")
    print(f"User prompt: {user_length:,} chars (~{user_length // 4:,} tokens)")
    print(f"Total: {total_length:,} chars (~{estimated_tokens:,} tokens)")
    print(f"Questions included: {num_questions}")
    
    if num_questions < 50:
        full_estimated = estimated_tokens * (50 / num_questions)
        print(f"Full 50 questions estimated: ~{int(full_estimated):,} tokens")
    
    print("\n✅ Prompt generation successful!")
    print("="*80)


def compare_versions(strategy='random', num_questions=5):
    """Compare v1 and v2 prompt structures side by side"""
    
    print("\n" + "="*80)
    print("COMPARING V1 vs V2 PROMPTS")
    print("="*80 + "\n")
    
    for version in ['v1', 'v2']:
        test_prompt(version=version, strategy=strategy, num_questions=num_questions)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test prompt generation for v1 and v2 personas")
    parser.add_argument('--version', type=str, default='v2', choices=['v1', 'v2', 'both'],
                       help='Persona version to test (default: v2)')
    parser.add_argument('--strategy', type=str, default='random', choices=['random', 'ipip', 'wpp'],
                       help='Persona strategy to use (default: random)')
    parser.add_argument('--questions', type=int, default=5,
                       help='Number of questions to include (default: 5)')
    
    args = parser.parse_args()
    
    if args.version == 'both':
        compare_versions(strategy=args.strategy, num_questions=args.questions)
    else:
        test_prompt(version=args.version, strategy=args.strategy, num_questions=args.questions)
