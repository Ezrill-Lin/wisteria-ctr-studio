"""PersonalitySamplingAgent Package

This package provides tools for generating and testing personality profiles using
the Big Five personality traits (OCEAN). It includes functionality for creating
diverse, realistic personality profiles and testing them with personality questionnaires.

Main Components:
- personality_sampling_agent: Core personality generation and testing functionality
- Personality_profiles/: Generated personality profile storage directory

Features:
- Generate diverse personality profiles using LLM
- Test personality consistency with questionnaires
- Export profiles for use in other applications
- Big Five (OCEAN) personality trait scoring
"""

from .personality_sampling_agent import (
    generate_personality_profiles,
    test_personality_profiles,
    PersonalityGenerator
)

__version__ = "1.0.0"
__all__ = [
    "generate_personality_profiles",
    "test_personality_profiles", 
    "PersonalityGenerator"
]