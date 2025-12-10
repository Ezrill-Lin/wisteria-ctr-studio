"""
Test script to check the prompt format for CTR prediction personas

This script helps you inspect:
- System messages with persona embodiment
- User prompts for ad evaluation
- Token estimates for different configurations
"""

import json
import argparse
from pathlib import Path
from utils import (
    get_persona_file_path,
    load_personas_from_file,
    create_persona_system_message,
    create_persona_user_prompt,
    create_image_ad_prompt
)


def test_text_ad_prompt(
    version='v2',
    strategy='random',
    ad_text="Get 50% Off Premium Coffee - Limited Time Offer!",
    platform='facebook',
    realistic_mode=True
):
    """Test prompt generation for text-based ad
    
    Args:
        version: Persona version ('v1' or 'v2')
        strategy: Strategy name ('random', 'wpp', 'ipip')
        ad_text: Advertisement text to evaluate
        platform: Ad platform (facebook, instagram, tiktok, youtube, amazon)
        realistic_mode: Use enhanced real-world browsing context
    """
    
    # Load one persona
    persona_file = get_persona_file_path(version, strategy)
    personas = load_personas_from_file(persona_file, sample_size=1)
    persona = personas[0]
    
    # Create prompts
    system_message = create_persona_system_message(persona, version)
    user_prompt = create_persona_user_prompt(ad_text, platform, realistic_mode=realistic_mode)
    
    # Print results
    print("="*80)
    mode_label = "REALISTIC" if realistic_mode else "SIMPLE"
    print(f"TEXT AD PROMPT TEST ({mode_label} MODE) - {version.upper()} PERSONA ({strategy.upper()} STRATEGY)")
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
        if 'self_schema' in persona and persona['self_schema']:
            print(f"Self-Schema: {len(persona['self_schema'])} core beliefs")
            for belief in persona['self_schema'][:3]:  # Show first 3
                print(f"    • {belief}")
    
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
    print(f"Platform: {platform}")
    print(f"Ad text length: {len(ad_text)} chars")
    
    print("\n✅ Text ad prompt generation successful!")
    print("="*80)


def test_image_ad_prompt(
    version='v2',
    strategy='random',
    image_url="https://example.com/ad-image.jpg",
    platform='facebook'
):
    """Test prompt generation for image-based ad
    
    Args:
        version: Persona version ('v1' or 'v2')
        strategy: Strategy name ('random', 'wpp', 'ipip')
        image_url: URL or path to advertisement image
        platform: Ad platform (facebook, instagram, tiktok, youtube, amazon)
    """
    
    # Load one persona
    persona_file = get_persona_file_path(version, strategy)
    personas = load_personas_from_file(persona_file, sample_size=1)
    persona = personas[0]
    
    # Create prompts
    system_message = create_persona_system_message(persona, version)
    user_prompt_text = create_image_ad_prompt(image_url, platform)
    
    # Print results
    print("="*80)
    print(f"IMAGE AD PROMPT TEST - {version.upper()} PERSONA ({strategy.upper()} STRATEGY)")
    print("="*80)
    
    print("\n📋 PERSONA INFO:")
    print(f"ID: {persona.get('id', 'N/A')}")
    print(f"Demographics: {persona['demographics']}")
    
    if version == 'v2':
        print("\n🎯 V2 ADDITIONAL FIELDS:")
        if 'behavioral_tendencies' in persona:
            print(f"Behavioral Tendencies: {len(persona['behavioral_tendencies'])} categories")
        if 'self_schema' in persona and persona['self_schema']:
            print(f"Self-Schema: {len(persona['self_schema'])} core beliefs")
    
    print("\n📝 SYSTEM MESSAGE:")
    print("-"*80)
    print(system_message)
    print("-"*80)
    
    print("\n📝 USER PROMPT (Image Ad):")
    print("-"*80)
    print(user_prompt_text)
    print("-"*80)
    
    print("\n📊 PROMPT STATISTICS:")
    system_length = len(system_message)
    user_length = len(user_prompt_text)
    total_length = system_length + user_length
    estimated_tokens = total_length // 4
    
    print(f"System message: {system_length:,} chars (~{system_length // 4:,} tokens)")
    print(f"User prompt: {user_length:,} chars (~{user_length // 4:,} tokens)")
    print(f"Total: {total_length:,} chars (~{estimated_tokens:,} tokens)")
    print(f"Platform: {platform}")
    print(f"Image URL: {image_url}")
    print(f"Note: Actual token count includes image tokens (varies by image size)")
    
    print("\n✅ Image ad prompt generation successful!")
    print("="*80)


def compare_versions(
    strategy='random',
    ad_text="Get 50% Off Premium Coffee - Limited Time Offer!",
    platform='facebook'
):
    """Compare v1 and v2 prompt structures side by side"""
    
    print("\n" + "="*80)
    print("COMPARING V1 vs V2 PROMPTS FOR CTR PREDICTION")
    print("="*80 + "\n")
    
    for version in ['v1', 'v2']:
        test_text_ad_prompt(
            version=version,
            strategy=strategy,
            ad_text=ad_text,
            platform=platform
        )
        print()


def test_multiple_platforms(version='v2', strategy='random'):
    """Test how prompts change across different platforms"""
    
    platforms = ['facebook', 'instagram', 'tiktok', 'youtube', 'amazon']
    ad_text = "New iPhone 15 Pro - Pre-Order Now and Save $200!"
    
    print("\n" + "="*80)
    print("TESTING PLATFORM-SPECIFIC CONTEXT")
    print("="*80 + "\n")
    
    # Load one persona
    persona_file = get_persona_file_path(version, strategy)
    personas = load_personas_from_file(persona_file, sample_size=1)
    persona = personas[0]
    
    print(f"Using {version.upper()} persona: {persona.get('id', 'N/A')}")
    print(f"Ad text: {ad_text}\n")
    
    for platform in platforms:
        user_prompt = create_persona_user_prompt(ad_text, platform)
        
        print(f"\n{'─'*80}")
        print(f"PLATFORM: {platform.upper()}")
        print('─'*80)
        
        # Extract just the context line
        lines = user_prompt.split('\n')
        for line in lines:
            if line.startswith('Context:'):
                print(line)
                break
        
        print(f"Prompt length: {len(user_prompt)} chars (~{len(user_prompt) // 4} tokens)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test prompt generation for CTR prediction personas"
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v2',
        choices=['v1', 'v2', 'both'],
        help='Persona version to test (default: v2)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='random',
        choices=['random', 'ipip', 'wpp'],
        help='Persona strategy to use (default: random)'
    )
    parser.add_argument(
        '--ad-type',
        type=str,
        default='text',
        choices=['text', 'image', 'platforms'],
        help='Type of ad to test (default: text)'
    )
    parser.add_argument(
        '--ad-text',
        type=str,
        default="Get 50% Off Premium Coffee - Limited Time Offer!",
        help='Advertisement text for testing'
    )
    parser.add_argument(
        '--image-url',
        type=str,
        default="https://example.com/ad-image.jpg",
        help='Image URL for image ad testing'
    )
    parser.add_argument(
        '--platform',
        type=str,
        default='facebook',
        choices=['facebook', 'instagram', 'tiktok', 'youtube', 'amazon'],
        help='Ad platform (default: facebook)'
    )
    parser.add_argument(
        '--realistic-mode',
        action='store_true',
        default=False,
        help='Use realistic browsing context mode (default: True in production)'
    )
    parser.add_argument(
        '--simple-mode',
        action='store_true',
        default=False,
        help='Use simple evaluation mode (for comparison)'
    )
    
    args = parser.parse_args()
    
    # Determine realistic_mode based on flags
    if args.simple_mode:
        realistic_mode = False
    elif args.realistic_mode:
        realistic_mode = True
    else:
        realistic_mode = True  # Default to realistic mode
    
    if args.version == 'both':
        compare_versions(
            strategy=args.strategy,
            ad_text=args.ad_text,
            platform=args.platform
        )
    elif args.ad_type == 'image':
        test_image_ad_prompt(
            version=args.version,
            strategy=args.strategy,
            image_url=args.image_url,
            platform=args.platform
        )
    elif args.ad_type == 'platforms':
        test_multiple_platforms(
            version=args.version,
            strategy=args.strategy
        )
    else:
        test_text_ad_prompt(
            version=args.version,
            strategy=args.strategy,
            ad_text=args.ad_text,
            platform=args.platform,
            realistic_mode=realistic_mode
        )
