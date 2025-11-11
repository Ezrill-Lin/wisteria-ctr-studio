"""Main script to run PersonaGenerationAgent

This script provides a command-line interface for generating personas
from demographic data and personality profiles.
"""

import asyncio
import argparse
from PersonaGenerationAgent import PersonaGenerationAgent


async def main():
    """Main function to run persona generation."""
    parser = argparse.ArgumentParser(description="Generate personas from demographics and personality profiles")
    parser.add_argument(
        "--demographics",
        type=str,
        default="data/synthetic_demographics_1m.parquet",
        help="Path to demographic data parquet file"
    )
    parser.add_argument(
        "--personalities",
        type=str,
        default="PersonalitySamplingAgent/Personality_profiles/profiles.json",
        help="Path to personality profiles JSON file"
    )
    parser.add_argument(
        "--num-personas",
        type=int,
        default=100,
        help="Number of personas to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_personas.json",
        help="Output path for generated personas"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=30,
        help='Maximum number of concurrent API requests (default: 30, recommended: 10-50 based on API tier and network capacity)'
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    print("Initializing PersonaGenerationAgent...")
    agent = PersonaGenerationAgent(model=args.model)
    
    # Load data
    print("Loading data...")
    demographics = agent.load_demographic_data(args.demographics)
    personalities = agent.load_personality_profiles(args.personalities)
    
    # Generate personas asynchronously
    print(f"\nGenerating {args.num_personas} personas...")
    personas = await agent.generate_personas(
        demographic_data=demographics,
        personality_profiles=personalities,
        num_personas=args.num_personas,
        output_path=args.output,
        random_seed=args.seed,
        max_concurrent=args.max_concurrent
    )
    
    # Display sample personas
    print("\n" + "="*80)
    print("SAMPLE GENERATED PERSONAS")
    print("="*80)
    
    for i in range(min(3, len(personas))):
        persona = personas[i]
        print(f"\nPersona #{persona['id']}:")
        print("-" * 80)
        print(persona['description'])
        print()


if __name__ == "__main__":
    asyncio.run(main())
