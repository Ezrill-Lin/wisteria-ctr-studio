"""
Display and output formatting utilities for CTR prediction results.

This module contains helper functions for:
- Pretty-printing prediction results
- Formatting output to console
- Saving results to JSON files
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CTRPrediction import CTRPredictionResult


def print_separator(char="=", length=80):
    """Print a separator line.
    
    Args:
        char: Character to use for the separator
        length: Length of the separator line
    """
    print(char * length)


def print_result(result: 'CTRPredictionResult'):
    """Pretty-print the CTR prediction result to console.
    
    Args:
        result: CTRPredictionResult object containing prediction data
    """
    print_separator()
    print("CTR PREDICTION RESULTS")
    print_separator()
    print(f"\n📊 Overall Metrics")
    print(f"   Predicted CTR: {result.ctr:.2%}")
    print(f"   Total Clicks: {result.total_clicks:,} / {result.total_personas:,}")
    print(f"   Provider: {result.provider_used}")
    print(f"   Model: {result.model_used}")
    print(f"   Platform: {result.ad_platform}")
    
    print(f"\n✅ Personas Who WOULD Click ({sum(1 for r in result.persona_responses if r.will_click)}):")
    print_separator("-", 80)
    
    click_count = 0
    for resp in result.persona_responses:
        if resp.will_click and click_count < 5:  # Show first 5
            print(f"\n   Persona {resp.persona_id}")
            if resp.demographics:
                print(f"   Demographics: {resp.demographics}")
            print(f"   Reasoning: {resp.reasoning}")
            click_count += 1
    
    if sum(1 for r in result.persona_responses if r.will_click) > 5:
        print(f"\n   ... and {sum(1 for r in result.persona_responses if r.will_click) - 5} more")
    
    print(f"\n❌ Personas Who WOULD NOT Click ({sum(1 for r in result.persona_responses if not r.will_click)}):")
    print_separator("-", 80)
    
    no_click_count = 0
    for resp in result.persona_responses:
        if not resp.will_click and no_click_count < 5:  # Show first 5
            print(f"\n   Persona {resp.persona_id}")
            if resp.demographics:
                print(f"   Demographics: {resp.demographics}")
            print(f"   Reasoning: {resp.reasoning}")
            no_click_count += 1
    
    if sum(1 for r in result.persona_responses if not r.will_click) > 5:
        print(f"\n   ... and {sum(1 for r in result.persona_responses if not r.will_click) - 5} more")
    
    print(f"\n\n📝 FINAL ANALYSIS")
    print_separator("=", 80)
    print(result.final_analysis)
    print_separator("=", 80)


def save_result_json(result: 'CTRPredictionResult', output_path: str):
    """Save CTR prediction result to a JSON file.
    
    Args:
        result: CTRPredictionResult object containing prediction data
        output_path: Path where JSON file should be saved
    """
    data = {
        "ctr": result.ctr,
        "total_personas": result.total_personas,
        "total_clicks": result.total_clicks,
        "provider_used": result.provider_used,
        "model_used": result.model_used,
        "ad_platform": result.ad_platform,
        "persona_responses": [
            {
                "persona_id": r.persona_id,
                "will_click": r.will_click,
                "reasoning": r.reasoning,
                "demographics": r.demographics,
                "ocean_scores": r.ocean_scores
            }
            for r in result.persona_responses
        ],
        "final_analysis": result.final_analysis
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
