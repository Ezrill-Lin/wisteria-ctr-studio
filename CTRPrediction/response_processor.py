"""
Response Processor Module

Handles aggregation and analysis of persona responses for CTR predictions.
"""

import os
from typing import List

from .utils import create_analysis_prompt


async def generate_final_analysis(
    ad_content: str,
    ad_platform: str,
    persona_responses: List,
    ctr: float
) -> str:
    """Generate final analysis from all persona responses using Gemini.
    
    Args:
        ad_content: Original ad content (text or image description)
        ad_platform: Platform where ad is shown
        persona_responses: List of PersonaResponse objects
        ctr: Computed CTR
        
    Returns:
        Analysis text with insights and recommendations
    """
    # Prepare summary of responses
    click_reasons = [resp.reasoning for resp in persona_responses if resp.will_click]
    no_click_reasons = [resp.reasoning for resp in persona_responses if not resp.will_click]
    
    # Create analysis prompt
    system_message, user_prompt = create_analysis_prompt(
        ad_content=ad_content,
        ad_platform=ad_platform,
        total_personas=len(persona_responses),
        total_clicks=sum(1 for r in persona_responses if r.will_click),
        ctr=ctr,
        click_reasons=click_reasons,
        no_click_reasons=no_click_reasons
    )
    
    # Call Gemini for analysis
    try:
        from google import genai
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Combine system message and user prompt
        full_prompt = f"{system_message}\n\n{user_prompt}"
        
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=5000
            )
        )
        analysis = response.text
        
    except Exception as e:
        analysis = f"[Error generating analysis: {type(e).__name__}: {e}]"
    
    return analysis
