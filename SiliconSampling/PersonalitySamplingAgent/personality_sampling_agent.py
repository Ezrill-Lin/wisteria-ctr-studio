"""Personality Sampling Agent

This module provides functionality for generating and testing personality profiles
using the Big Five personality traits (OCEAN) through LLM interactions.
"""

from openai import OpenAI 
from tqdm import tqdm
import os
import json
import argparse
from typing import Dict, List, Optional


class PersonalityGenerator:
    """A class for generating and testing personality profiles using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the personality generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            base_url: Custom API base URL (optional)
        """
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or "https://api.chatanywhere.tech/v1"
        )
    
    def generate_profiles(self, num_profiles: int, prompt_file: str, output_dir: str = "Personality_profiles") -> Dict:
        """Generate personality profiles using LLM.
        
        Args:
            num_profiles: Number of personality profiles to generate
            prompt_file: Path to the personality generation prompt file
            output_dir: Directory to save generated profiles
            
        Returns:
            Dictionary containing generated personality profiles
        """
        os.makedirs(output_dir, exist_ok=True)
        
        prefix = f"""
        You are a personality profiler. You are doing a personality generation job for simulating humans.

        ### Tasks
        Your task is to use the Big Five personality traits (OCEAN) defined below, and produce {num_profiles} different detailed personality description plus scores for each dimension.
        You should pay attention to the diversity among different profiles. Also make sure each profile is realistic and coherent.
        """

        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt = file.read()

        prompt = prefix + prompt

        print(f"Generating {num_profiles} personality profiles...")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        json_result = response.choices[0].message.content
        profiles = json.loads(json_result)
        
        # Save profiles to file
        with open(os.path.join(output_dir, "profiles.json"), "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, "profiles.txt"), "w", encoding="utf-8") as f:
            f.write(json_result)
        
        return profiles

    def test_profiles(self, profiles: Dict, questions_file: str, output_dir: str = "Personality_profiles") -> None:
        """Test personality profiles with questionnaire.
        
        Args:
            profiles: Dictionary of personality profiles to test
            questions_file: Path to file containing personality questions
            output_dir: Directory to save test results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = f.readlines()
            questions = [q.strip() for q in questions if q.strip()]
        
        num_profiles = len([k for k in profiles.keys() if k.startswith('profile_')])
        
        for j in range(num_profiles):
            profile_key = f'profile_{j+1}'
            if profile_key not in profiles:
                continue
                
            profile = profiles[profile_key]
            print(f"Testing profile {j+1} with {len(questions)} questions...")
            
            test_file = os.path.join(output_dir, f"profile_{j+1}_test.txt")
            
            for i, question in tqdm(enumerate(questions), desc=f"Profile {j+1}"):
                test_prompt = f"""
                You are given the simulated personality profile of a person based on the Big Five Personality traits (OCEAN).
                Your task is to answer the following QUESTION according to the personality profile provided.
                You should STAY CONSISTENT with the scores and trait descriptions.
                Do not invent biographical facts (no names, places, jobs, ages) unless explicitly provided.

                Big Five Personality traits scores with descriptions:
                    - Openness: {profile['Scores']['Openness']}
                    - Conscientiousness: {profile['Scores']['Conscientiousness']}
                    - Extraversion: {profile['Scores']['Extraversion']}
                    - Agreeableness: {profile['Scores']['Agreeableness']}
                    - Neuroticism: {profile['Scores']['Neuroticism']}
                Personality Description:
                    - {profile['Summary']}

                QUESTION:
                {question}

                Reply in first person. Describe how you would feel and what you would do in the situation. 
                Keep answers concise but natural in 3-4 sentences. Do not use bullet points.
                """

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Fixed model name
                    messages=[
                        {"role": "system", "content": test_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.1,
                ) 

                text = response.choices[0].message.content.strip()
                
                with open(test_file, "a", encoding="utf-8") as f:
                    f.write("\n\n")
                    f.write("Q: " + question + "\n")
                    f.write("A: " + text + "\n")


def generate_personality_profiles(num_profiles: int, prompt_file: str, 
                                output_dir: str = "Personality_profiles",
                                api_key: Optional[str] = None,
                                base_url: Optional[str] = None) -> Dict:
    """Generate personality profiles (standalone function).
    
    Args:
        num_profiles: Number of personality profiles to generate
        prompt_file: Path to the personality generation prompt file
        output_dir: Directory to save generated profiles
        api_key: OpenAI API key (optional, uses environment variable if not provided)
        base_url: Custom API base URL (optional)
        
    Returns:
        Dictionary containing generated personality profiles
    """
    generator = PersonalityGenerator(api_key=api_key, base_url=base_url)
    return generator.generate_profiles(num_profiles, prompt_file, output_dir)


def test_personality_profiles(profiles: Dict, questions_file: str,
                            output_dir: str = "Personality_profiles",
                            api_key: Optional[str] = None,
                            base_url: Optional[str] = None) -> None:
    """Test personality profiles with questionnaire (standalone function).
    
    Args:
        profiles: Dictionary of personality profiles to test
        questions_file: Path to file containing personality questions
        output_dir: Directory to save test results
        api_key: OpenAI API key (optional, uses environment variable if not provided)
        base_url: Custom API base URL (optional)
    """
    generator = PersonalityGenerator(api_key=api_key, base_url=base_url)
    generator.test_profiles(profiles, questions_file, output_dir)


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
    return parser.parse_args()


def main():
    """Main function for command line usage."""
    args = parse_args()
    
    # Initialize the personality generator
    generator = PersonalityGenerator(api_key=args.api_key, base_url=args.base_url)
    
    # Generate profiles
    profiles = generator.generate_profiles(
        num_profiles=args.num_profiles,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir
    )
    
    # Test profiles if requested
    if args.test:
        print('Testing personality profiles...')
        generator.test_profiles(
            profiles=profiles,
            questions_file=args.questions_file,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()