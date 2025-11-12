"""Persona Generation Agent

This module combines demographic data from census synthesis with personality profiles
to generate complete persona descriptions in a natural, descriptive style.
"""

import os
import json
import pandas as pd
from openai import AsyncOpenAI
from typing import Dict, List, Optional
from tqdm import tqdm
import random
import asyncio


class PersonaGenerationAgent:
    """Generate descriptive persona narratives combining demographics and personality.
    
    Uses async/await for high-performance concurrent persona generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the persona generation agent.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: OpenAI model to use for generation
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        
    def load_demographic_data(self, parquet_path: str) -> pd.DataFrame:
        """Load synthetic demographic data from parquet file.
        
        Args:
            parquet_path: Path to the synthetic demographics parquet file
            
        Returns:
            DataFrame containing demographic data
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Demographic data file not found: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} demographic records from {parquet_path}")
        return df
    
    def load_personality_profiles(self, profiles_path: str) -> List[Dict]:
        """Load personality profiles from JSON file.
        
        Args:
            profiles_path: Path to the personality profiles JSON file
            
        Returns:
            List of personality profile dictionaries
        """
        if not os.path.exists(profiles_path):
            raise FileNotFoundError(f"Personality profiles file not found: {profiles_path}")
        
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        print(f"Loaded {len(profiles)} personality profiles from {profiles_path}")
        return profiles
    
    def format_demographic_info(self, demographic: pd.Series) -> str:
        """Format demographic information into a readable string.
        
        Args:
            demographic: Series containing demographic data
            
        Returns:
            Formatted string describing demographics
        """
        demo_parts = []
        
        # Age and Gender
        demo_parts.append(f"Age: {demographic.get('age', 'Unknown')}")
        demo_parts.append(f"Gender: {demographic.get('gender', 'Unknown')}")
        
        # State
        if 'state' in demographic:
            demo_parts.append(f"Location: {demographic['state']}")
        
        # Race
        if 'race' in demographic:
            demo_parts.append(f"Race/Ethnicity: {demographic['race']}")
        
        # Education
        if 'educational_attainment' in demographic:
            demo_parts.append(f"Education: {demographic['educational_attainment']}")
        
        # Occupation
        if 'occupation' in demographic:
            demo_parts.append(f"Occupation: {demographic['occupation']}")
        
        return "\n".join(demo_parts)
    
    def format_personality_info(self, personality: Dict) -> str:
        """Format personality information into a readable string.
        
        Args:
            personality: Dictionary containing personality profile
            
        Returns:
            Formatted string describing personality traits
        """
        # Extract Big Five scores if available
        if 'scores' in personality:
            scores = personality['scores']
            personality_parts = [
                "Big Five Personality Traits:",
                f"- Openness: {scores.get('openness', 'N/A')}/10",
                f"- Conscientiousness: {scores.get('conscientiousness', 'N/A')}/10",
                f"- Extraversion: {scores.get('extraversion', 'N/A')}/10",
                f"- Agreeableness: {scores.get('agreeableness', 'N/A')}/10",
                f"- Neuroticism: {scores.get('neuroticism', 'N/A')}/10"
            ]
        else:
            personality_parts = []
        
        # Add description if available
        if 'description' in personality:
            personality_parts.append(f"\nPersonality Description:\n{personality['description']}")
        
        return "\n".join(personality_parts)
    
    async def generate_persona_description(self, demographic: pd.Series, personality: Dict) -> str:
        """Generate a descriptive persona paragraph using LLM (async).
        
        Args:
            demographic: Series containing demographic data
            personality: Dictionary containing personality profile
            
        Returns:
            Generated persona description starting with "You are..."
        """
        demographic_info = self.format_demographic_info(demographic)
        personality_info = self.format_personality_info(personality)
        
        prompt = f"""You are a persona creation expert. Based on the following demographic and personality information, create a coherent, natural-sounding persona description in second-person perspective.

DEMOGRAPHIC INFORMATION:
{demographic_info}

PERSONALITY INFORMATION:
{personality_info}

TASK:
Write a single, flowing paragraph (3-5 sentences) that describes this person in second-person ("You are..."). The description should:
1. Start with "You are"
2. Seamlessly integrate demographic details (age, location, occupation, education)
3. Naturally weave in personality traits without explicitly mentioning the Big Five dimensions
4. Feel authentic and realistic
5. Avoid stereotypes or clichés
6. Be conversational and engaging

Example style:
"You are a 34-year-old software engineer living in Seattle, Washington. With a master's degree in computer science, you approach your work with meticulous attention to detail and a genuine curiosity about emerging technologies. Though you prefer deep, meaningful conversations with a small circle of friends over large social gatherings, you're always eager to collaborate on innovative projects that challenge the status quo. Your analytical mind and calm demeanor make you a trusted problem-solver, though you sometimes overthink decisions in your personal life."

Now generate the persona description:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a skilled persona writer who creates authentic, engaging character descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating persona description: {e}")
            return f"You are a {demographic.get('age', 'N/A')}-year-old {demographic.get('gender', 'person')} from {demographic.get('state', 'Unknown')}."
    
    async def generate_personas(
        self, 
        demographic_data: pd.DataFrame,
        personality_profiles: List[Dict],
        num_personas: int = 100,
        output_path: str = "generated_personas.json",
        random_seed: int = 42,
        max_concurrent: int = 10
    ) -> List[Dict]:
        """Generate complete personas asynchronously with concurrent API calls.
        
        Args:
            demographic_data: DataFrame containing demographic data
            personality_profiles: List of personality profile dictionaries
            num_personas: Number of personas to generate
            output_path: Path to save generated personas
            random_seed: Random seed for reproducibility
            max_concurrent: Maximum number of concurrent API calls
            
        Returns:
            List of generated persona dictionaries
        """
        random.seed(random_seed)
        
        # Sample demographics and personalities
        sampled_demographics = demographic_data.sample(n=num_personas, random_state=random_seed)
        
        # If we have fewer personality profiles than needed, cycle through them
        sampled_personalities = []
        for i in range(num_personas):
            personality_idx = i % len(personality_profiles)
            sampled_personalities.append(personality_profiles[personality_idx])
        
        async def generate_single_persona(idx: int, demographic: pd.Series, personality: Dict) -> Dict:
            """Generate a single persona asynchronously."""
            description = await self.generate_persona_description(demographic, personality)
            
            return {
                "id": idx + 1,
                "demographics": {
                    "age": int(demographic.get('age', 0)),
                    "gender": str(demographic.get('gender', '')),
                    "state": str(demographic.get('state', '')),
                    "race": str(demographic.get('race', '')),
                    "educational_attainment": str(demographic.get('educational_attainment', '')),
                    "occupation": str(demographic.get('occupation', ''))
                },
                "personality": personality,
                "description": description
            }
        
        # Create tasks for all personas
        tasks = []
        for idx, (demo_idx, demographic) in enumerate(sampled_demographics.iterrows()):
            personality = sampled_personalities[idx]
            tasks.append(generate_single_persona(idx, demographic, personality))
        
        # Process tasks with concurrency limit using semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_generate(task):
            async with semaphore:
                return await task
        
        print(f"\nGenerating {num_personas} personas asynchronously (max {max_concurrent} concurrent)...")
        
        # Execute all tasks with progress bar
        personas = []
        completed_tasks = asyncio.as_completed([bounded_generate(task) for task in tasks])
        
        with tqdm(total=num_personas, desc="Generating personas") as pbar:
            for coro in completed_tasks:
                persona = await coro
                personas.append(persona)
                pbar.update(1)
        
        # Sort personas by ID to maintain order
        personas.sort(key=lambda x: x['id'])
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Generated {len(personas)} personas")
        print(f"✓ Saved to: {output_path}")
        
        return personas

