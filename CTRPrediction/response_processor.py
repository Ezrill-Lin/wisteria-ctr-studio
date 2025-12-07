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
    """Generate final analysis from all persona responses using DeepSeek.
    
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
    
    # Call DeepSeek for analysis
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        )
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        analysis = response.choices[0].message.content
        
    except Exception as e:
        analysis = f"[Error generating analysis: {type(e).__name__}: {e}]"
    
    return analysis
