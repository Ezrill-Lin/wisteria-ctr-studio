"""Main script for Personality Sampling Agent

This script provides a command-line interface for generating and testing
personality profiles using the Big Five personality traits.
"""

import asyncio
import argparse
from personality_sampling_agent import PersonalityGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Personality Sampling Agent")
    parser.add_argument("--num_profiles", type=int, default=2,
                       help="Number of personality profiles to generate and test")
    parser.add_argument("--questions_file", type=str, default="personality_questions.txt",
                       help="Path to the file containing personality questions")
    parser.add_argument("--prompt_file", type=str, default="10.25_big5_prompt.txt",
                       help="Path to the personality generation prompt file")
    parser.add_argument("--test", action="store_true", default=False,
                       help="Whether to run in test mode with personality questionnaire")
    parser.add_argument("--output_dir", type=str, default="Personality_profiles",
                       help="Directory to save generated profiles and test results")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)")
    parser.add_argument("--base_url", type=str, default=None,
                       help="Custom API base URL (optional)")
    parser.add_argument("--max_concurrent", type=int, default=30,
                       help="Maximum number of concurrent API requests (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (omit for random generation)")
    return parser.parse_args()


async def main():
    """Main function for command line usage."""
    args = parse_args()
    
    # Initialize the personality generator
    generator = PersonalityGenerator(api_key=args.api_key, base_url=args.base_url)
    
    # Generate profiles
    profiles = await generator.generate_profiles(
        num_profiles=args.num_profiles,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        seed=args.seed
    )
    
    # Test profiles if requested
    if args.test:
        print('Testing personality profiles...')
        await generator.test_profiles(
            profiles=profiles,
            questions_file=args.questions_file,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    asyncio.run(main())
