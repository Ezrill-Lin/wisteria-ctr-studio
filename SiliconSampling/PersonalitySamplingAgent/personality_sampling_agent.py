"""Personality Sampling Agent

This module provides functionality for generating and testing personality profiles
using the Big Five personality traits (OCEAN) through LLM interactions.
"""

from openai import AsyncOpenAI 
from tqdm import tqdm
import os
import json
import asyncio
from typing import Dict, List, Optional


class PersonalityGenerator:
    """A class for generating and testing personality profiles using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the personality generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            base_url: Custom API base URL (optional)
        """
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url  # Don't set default base_url, use OpenAI's default
        )
        self._prompt_cache = {}  # Cache for loaded prompts
    
    async def generate_single_profile(self, profile_id: int, prompt_file: str, seed: Optional[int] = None) -> Dict:
        """Generate a single personality profile using LLM.
        
        Args:
            profile_id: ID number for this profile
            prompt_file: Path to the personality generation prompt file
            seed: Random seed for reproducibility (None = random generation)
            
        Returns:
            Dictionary containing a single personality profile
        """
        # Cache the prompt to avoid reading file multiple times
        if prompt_file not in self._prompt_cache:
            with open(prompt_file, "r", encoding="utf-8") as file:
                self._prompt_cache[prompt_file] = file.read()
        
        base_prompt = self._prompt_cache[prompt_file]

        # Simplified prompt - just the essentials
        prompt = f"""Generate a realistic Big Five (OCEAN) personality profile with random scores (0-10 integers) and a coherent description.

Output JSON format:
{{
    "scores": {{
        "openness": <0-10 integer>,
        "conscientiousness": <0-10 integer>,
        "extraversion": <0-10 integer>,
        "agreeableness": <0-10 integer>,
        "neuroticism": <0-10 integer>
    }},
    "description": "<one detailed paragraph describing how these scores manifest in behavior>"
}}

Make the profile diverse and realistic. Vary the scores significantly across profiles."""

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.9,  # Higher temperature for more diversity
            response_format={"type": "json_object"},
            seed=seed  # Use seed if provided for reproducibility
        )
        
        json_result = response.choices[0].message.content
        profile = json.loads(json_result)
        
        # Add the ID manually to ensure correct format
        profile["id"] = profile_id
        
        return profile
    
    async def generate_profiles(self, num_profiles: int, prompt_file: str, output_dir: str = "Personality_profiles", max_concurrent: int = 30, seed: Optional[int] = None) -> List[Dict]:
        """Generate personality profiles using LLM with concurrent API calls.
        
        Args:
            num_profiles: Number of personality profiles to generate
            prompt_file: Path to the personality generation prompt file
            output_dir: Directory to save generated profiles
            max_concurrent: Maximum number of concurrent API calls
            seed: Random seed for reproducibility (None = random generation)
            
        Returns:
            List of generated personality profile dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {num_profiles} personality profiles asynchronously (max {max_concurrent} concurrent)...")
        if seed is not None:
            print(f"Using seed: {seed} for reproducibility")
        
        # Create tasks for all profiles
        tasks = []
        for i in range(num_profiles):
            tasks.append(self.generate_single_profile(i + 1, prompt_file, seed))
        
        # Process tasks with concurrency limit using semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_generate(task):
            async with semaphore:
                return await task
        
        # Execute all tasks with progress bar
        profiles = []
        completed_tasks = asyncio.as_completed([bounded_generate(task) for task in tasks])
        
        with tqdm(total=num_profiles, desc="Generating profiles") as pbar:
            for coro in completed_tasks:
                profile = await coro
                profiles.append(profile)
                pbar.update(1)
        
        # Sort profiles by ID
        profiles.sort(key=lambda x: x.get('id', 0))
        
        # Load existing profiles if file exists and append new ones
        profiles_file = os.path.join(output_dir, "profiles.json")
        existing_profiles = []
        if os.path.exists(profiles_file):
            with open(profiles_file, 'r', encoding='utf-8') as f:
                existing_profiles = json.load(f)
            print(f"Found {len(existing_profiles)} existing profiles")
            
            # Renumber new profiles to continue from existing max ID
            if existing_profiles:
                max_existing_id = max(p.get('id', 0) for p in existing_profiles)
                for profile in profiles:
                    profile['id'] = max_existing_id + profile['id']
            
            # Append new profiles to existing ones
            all_profiles = existing_profiles + profiles
        else:
            all_profiles = profiles
        
        # Save all profiles to file
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Generated {len(profiles)} new profiles")
        print(f"✓ Total profiles in file: {len(all_profiles)}")
        print(f"✓ Saved to: {profiles_file}")
        
        return profiles

    async def test_profiles(self, profiles: List[Dict], questions_file: str, output_dir: str = "Personality_profiles") -> None:
        """Test personality profiles with questionnaire.
        
        Args:
            profiles: List of personality profiles to test
            questions_file: Path to file containing personality questions
            output_dir: Directory to save test results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = f.readlines()
            questions = [q.strip() for q in questions if q.strip()]
        
        for profile in profiles:
            profile_id = profile.get('id', 'unknown')
            scores = profile.get('scores', {})
            description = profile.get('description', '')
            
            print(f"Testing {profile_id} with {len(questions)} questions...")
            
            test_file = os.path.join(output_dir, f"{profile_id}_test.txt")
            
            for i, question in tqdm(enumerate(questions), desc=f"{profile_id}"):
                test_prompt = f"""
                You are given the simulated personality profile of a person based on the Big Five Personality traits (OCEAN).
                Your task is to answer the following QUESTION according to the personality profile provided.
                You should STAY CONSISTENT with the scores and trait descriptions.
                Do not invent biographical facts (no names, places, jobs, ages) unless explicitly provided.

                Big Five Personality traits scores (0-10 scale):
                    - Openness: {scores.get('openness', 'N/A')}/10
                    - Conscientiousness: {scores.get('conscientiousness', 'N/A')}/10
                    - Extraversion: {scores.get('extraversion', 'N/A')}/10
                    - Agreeableness: {scores.get('agreeableness', 'N/A')}/10
                    - Neuroticism: {scores.get('neuroticism', 'N/A')}/10
                Personality Description:
                    - {description}

                QUESTION:
                {question}

                Reply in first person. Describe how you would feel and what you would do in the situation. 
                Keep answers concise but natural in 3-4 sentences. Do not use bullet points.
                """

                response = await self.client.chat.completions.create(
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


async def generate_personality_profiles(num_profiles: int, prompt_file: str, 
                                output_dir: str = "Personality_profiles",
                                api_key: Optional[str] = None,
                                base_url: Optional[str] = None,
                                max_concurrent: int = 30,
                                seed: Optional[int] = None) -> List[Dict]:
    """Generate personality profiles (standalone function).
    
    Args:
        num_profiles: Number of personality profiles to generate
        prompt_file: Path to the personality generation prompt file
        output_dir: Directory to save generated profiles
        api_key: OpenAI API key (optional, uses environment variable if not provided)
        base_url: Custom API base URL (optional)
        max_concurrent: Maximum number of concurrent API calls
        seed: Random seed for reproducibility (None = random generation)
        
    Returns:
        List of generated personality profile dictionaries
    """
    generator = PersonalityGenerator(api_key=api_key, base_url=base_url)
    return await generator.generate_profiles(num_profiles, prompt_file, output_dir, max_concurrent, seed)


async def test_personality_profiles(profiles: List[Dict], questions_file: str,
                            output_dir: str = "Personality_profiles",
                            api_key: Optional[str] = None,
                            base_url: Optional[str] = None) -> None:
    """Test personality profiles with questionnaire (standalone function).
    
    Args:
        profiles: List of personality profiles to test
        questions_file: Path to file containing personality questions
        output_dir: Directory to save test results
        api_key: OpenAI API key (optional, uses environment variable if not provided)
        base_url: Custom API base URL (optional)
    """
    generator = PersonalityGenerator(api_key=api_key, base_url=base_url)
    await generator.test_profiles(profiles, questions_file, output_dir)
