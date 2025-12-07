"""
Utility functions for CTR prediction system.

This module contains helper functions for:
- Persona file path resolution
- Prompt construction
- Response parsing
- Demographics extraction
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_persona_file_path(persona_version: str, persona_strategy: str) -> Path:
    """Get the file path for a persona JSONL file.
    
    Args:
        persona_version: Version of personas ('v1' or 'v2')
        persona_strategy: Strategy for persona generation ('random', 'wpp', 'ipip')
        
    Returns:
        Path object pointing to the persona file
    """
    base_path = Path("SiliconSampling")
    
    if persona_version == "v2":
        return base_path / "personas_v2" / f"{persona_strategy}_matching" / f"personas_{persona_strategy}_v2.jsonl"
    else:
        return base_path / "personas" / f"{persona_strategy}_matching" / f"personas_{persona_strategy}.jsonl"


def load_personas_from_file(file_path: Path, sample_size: int) -> List[Dict[str, Any]]:
    """Load personas from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        sample_size: Number of personas to load
        
    Returns:
        List of persona dictionaries
        
    Raises:
        FileNotFoundError: If the persona file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Persona file not found: {file_path}\n"
            f"Please generate personas first."
        )
    
    personas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            persona = json.loads(line.strip())
            personas.append(persona)
            if len(personas) >= sample_size:
                break
    
    return personas


def create_persona_system_message(persona: Dict[str, Any], persona_version: str) -> str:
    """Create system message for persona embodiment.
    
    Args:
        persona: Persona dictionary
        persona_version: Version of personas ('v1' or 'v2')
        
    Returns:
        System message string
    """
    if persona_version == 'v1':
        # V1: Simple persona_description only
        return f"""<persona>
{persona['persona_description']}
</persona>

<instruction>
You are evaluating an advertisement. Embody this persona completely and decide whether you would click on this ad.
</instruction>"""
    
    # V2: Enhanced with behavioral_tendencies and self_schema
    system_message = f"""<persona>
{persona['persona_description']}
</persona>"""
    
    if 'behavioral_tendencies' in persona and persona['behavioral_tendencies']:
        system_message += "\n\n<behavioral_tendencies>\n"
        for key, value in persona['behavioral_tendencies'].items():
            system_message += f"- {key.replace('_', ' ').title()}: {value}\n"
        system_message += "</behavioral_tendencies>"
    
    if 'self_schema' in persona and persona['self_schema']:
        system_message += "\n\n<core_beliefs>\n"
        for belief in persona['self_schema']:
            system_message += f"- {belief}\n"
        system_message += "</core_beliefs>"
    
    system_message += "\n\n<instruction>\nYou are evaluating an advertisement. Embody this persona completely and decide whether you would click on this ad based on your characteristics, behaviors, and beliefs.\n</instruction>"
    
    return system_message


def create_persona_user_prompt(ad_content: str, ad_platform: str) -> str:
    """Create user prompt for ad evaluation.
    
    Args:
        ad_content: Advertisement content
        ad_platform: Platform where ad is shown
        
    Returns:
        User prompt string
    """
    platform_contexts = {
        "facebook": "This ad appears in your Facebook news feed while you browse social content.",
        "tiktok": "This ad appears between TikTok videos as you scroll through short-form content.",
        "amazon": "This ad appears on Amazon while you are shopping or browsing products.",
        "instagram": "This ad appears in your Instagram feed or stories.",
        "youtube": "This ad plays before or during a YouTube video you're watching."
    }
    
    platform_context = platform_contexts.get(ad_platform.lower(), platform_contexts["facebook"])
    
    return f"""Platform: {ad_platform}
Context: {platform_context}

Advertisement:
{ad_content}

Instructions:
1. Based on your persona (demographics, personality, behaviors, and beliefs), would you click on this ad?
2. Provide your decision and 1-3 sentences of reasoning.

Output your response in the following JSON format:
{{
  "will_click": true,  // or false
  "reasoning": "Your 1-3 sentence explanation here"
}}

Output ONLY valid JSON, nothing else."""


def create_analysis_prompt(
    ad_content: str,
    ad_platform: str,
    total_personas: int,
    total_clicks: int,
    ctr: float,
    click_reasons: List[str],
    no_click_reasons: List[str]
) -> Tuple[str, str]:
    """Create prompt for final analysis generation.
    
    Args:
        ad_content: Original ad content
        ad_platform: Platform
        total_personas: Total number of personas evaluated
        total_clicks: Number of personas who clicked
        ctr: Computed CTR
        click_reasons: List of reasons from personas who clicked
        no_click_reasons: List of reasons from personas who didn't click
        
    Returns:
        Tuple of (system_message, user_prompt)
    """
    system_message = """You are an expert marketing analyst specializing in advertisement performance evaluation.
Your task is to analyze CTR prediction results and provide actionable insights."""
    
    # Limit samples to 20 each
    click_sample = "\n".join([f"- {r}" for r in click_reasons[:20]])
    no_click_sample = "\n".join([f"- {r}" for r in no_click_reasons[:20]])
    
    user_prompt = f"""I ran a CTR prediction simulation on the following advertisement:

Platform: {ad_platform}
Advertisement Content:
{ad_content}

Prediction Results:
- Total Personas Evaluated: {total_personas}
- Predicted Clicks: {total_clicks}
- Predicted CTR: {ctr:.2%}

Sample Reasons for CLICKING (from personas who would click):
{click_sample if click_reasons else "None"}

Sample Reasons for NOT CLICKING (from personas who would not click):
{no_click_sample if no_click_reasons else "None"}

Based on this simulation, provide a comprehensive analysis covering:

1. **Overall Assessment**: Is the predicted CTR good, average, or poor for {ad_platform}? Why?

2. **What's Working Well**: What aspects of the ad are resonating with people who would click?

3. **What's Not Working**: What concerns or issues prevent people from clicking?

4. **Improvement Recommendations**: Provide 3-5 specific, actionable suggestions to improve this ad's CTR.

5. **Target Audience Insights**: Based on who clicked vs. who didn't, what can we learn about the ideal target audience?

Provide a well-structured analysis (use markdown formatting). Be specific and actionable."""
    
    return system_message, user_prompt


def parse_json_response(content: str) -> Tuple[bool, str]:
    """Parse JSON response from LLM.
    
    Args:
        content: Raw response content
        
    Returns:
        Tuple of (will_click, reasoning)
    """
    # Clean markdown code blocks if present
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    result = json.loads(content)
    will_click = result.get('will_click', False)
    reasoning = result.get('reasoning', 'No reasoning provided')
    
    return will_click, reasoning


def create_image_ad_prompt(ad_platform: str) -> str:
    """Create user prompt for image ad evaluation.
    
    Args:
        ad_platform: Platform where ad is shown
        
    Returns:
        User prompt string for image ad evaluation
    """
    platform_contexts = {
        "facebook": "This ad image appears in your Facebook news feed while you browse social content.",
        "tiktok": "This ad image appears between TikTok videos as you scroll through short-form content.",
        "amazon": "This ad image appears on Amazon while you are shopping or browsing products.",
        "instagram": "This ad image appears in your Instagram feed or stories.",
        "youtube": "This ad image appears before or during a YouTube video you're watching."
    }
    
    platform_context = platform_contexts.get(ad_platform.lower(), platform_contexts["facebook"])
    
    return f"""Platform: {ad_platform}
Context: {platform_context}

Instructions:
1. Look at this advertisement image carefully.
2. Based on your persona (demographics, personality, behaviors, and beliefs), would you click on this ad?
3. Provide your decision and 1-3 sentences of reasoning.

Output your response in the following JSON format:
{{
  "will_click": true,  // or false
  "reasoning": "Your 1-3 sentence explanation here"
}}

Output ONLY valid JSON, nothing else."""


def extract_demographics_string(demographics: Any) -> str:
    """Extract demographics as a readable string.
    
    Args:
        demographics: Demographics data (dict or string)
        
    Returns:
        Formatted demographics string
    """
    if isinstance(demographics, dict):
        return f"Age: {demographics.get('age', '?')}, Gender: {demographics.get('gender', '?')}, Occupation: {demographics.get('occupation', '?')}"
    return str(demographics)


def encode_image_to_data_url(image_path: str) -> str:
    """Encode a local image file to a data URL for vision API.
    
    Args:
        image_path: Path to local image file
        
    Returns:
        Data URL string (data:image/jpeg;base64,...)
        
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If file is not a valid image
    """
    import base64
    import mimetypes
    from pathlib import Path
    
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"File does not appear to be an image: {image_path}")
    
    # Read and encode image
    with open(path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{encoded_string}"
