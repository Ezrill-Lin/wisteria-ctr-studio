"""
CTR Predictor Module

Unified predictor for both text and image-based advertisements using GPT-4o-mini.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tqdm.asyncio import tqdm as async_tqdm

from .utils import (
    get_persona_file_path,
    load_personas_from_file,
    create_persona_system_message,
    create_persona_user_prompt,
    create_image_ad_prompt,
    parse_json_response,
    extract_demographics_string
)
from .response_processor import generate_final_analysis


@dataclass
class PersonaResponse:
    """Response from a single persona."""
    persona_id: str
    will_click: bool
    reasoning: str
    demographics: str = ""
    ocean_scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CTRPredictionResult:
    """Complete CTR prediction result."""
    ctr: float
    total_personas: int
    total_clicks: int
    persona_responses: List[PersonaResponse]
    final_analysis: str
    provider_used: str
    model_used: str
    ad_platform: str


class CTRPredictor:
    """Unified CTR predictor for text and image advertisements."""
    
    def __init__(
        self,
        persona_version: str = 'v2',
        persona_strategy: str = 'random',
        concurrent_requests: int = 10
    ):
        """Initialize CTR predictor.
        
        Uses gpt-4o-mini for individual click decisions and deepseek-chat for final analysis.
        
        Args:
            persona_version: Version of personas to use ('v1' or 'v2')
            persona_strategy: Strategy for persona generation ('random', 'wpp', 'ipip')
            concurrent_requests: Number of concurrent API calls
        """
        self.persona_version = persona_version
        self.persona_strategy = persona_strategy
        self.concurrent_requests = concurrent_requests
        
        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        print(f"✓ Initialized CTR Predictor")
        print(f"  Click Decision Model: gpt-4o-mini")
        print(f"  Analysis Model: deepseek-chat")
        print(f"  Persona Version: {persona_version}")
        print(f"  Persona Strategy: {persona_strategy}")
    
    def _load_personas(self, sample_size: int) -> List[Dict[str, Any]]:
        """Load personas from SiliconSampling folder."""
        persona_file = get_persona_file_path(self.persona_version, self.persona_strategy)
        
        print(f"Loading personas from {persona_file}...")
        personas = load_personas_from_file(persona_file, sample_size)
        
        print(f"✓ Loaded {len(personas):,} personas")
        return personas
    
    async def _get_persona_response_async(
        self,
        persona: Dict[str, Any],
        ad_text: Optional[str],
        image_url: Optional[str],
        ad_platform: str,
        semaphore: asyncio.Semaphore
    ) -> PersonaResponse:
        """Get click decision from a single persona.
        
        Args:
            persona: Persona data
            ad_text: Advertisement text (for text ads)
            image_url: Advertisement image URL (for image ads)
            ad_platform: Platform where ad is shown
            semaphore: Async semaphore for rate limiting
            
        Returns:
            PersonaResponse object
        """
        async with semaphore:
            system_message = create_persona_system_message(persona, self.persona_version)
            
            # Build user message based on ad type
            if image_url:
                # Image ad - use vision capabilities
                user_prompt = create_image_ad_prompt(ad_platform)
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            else:
                # Text ad
                user_prompt = create_persona_user_prompt(ad_text, ad_platform)
                user_content = user_prompt
            
            # Call GPT-4o-mini for click decisions with retry logic
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    from openai import AsyncOpenAI
                    
                    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    content = response.choices[0].message.content
                    
                    # Parse JSON response
                    will_click, reasoning = parse_json_response(content)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_type = type(e).__name__
                    
                    # Retry on rate limit or timeout errors
                    if attempt < max_retries - 1 and ('rate' in str(e).lower() or 'timeout' in str(e).lower()):
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    
                    # Final failure or non-retryable error
                    error_msg = f"{error_type}: {str(e)[:100]}"
                    print(f"Warning: Error processing persona {persona.get('id', '?')}: {error_msg}")
                    will_click = False
                    reasoning = "[API Error - response excluded from analysis]"
                    break
            
            # Extract demographics info
            demographics = extract_demographics_string(persona.get('demographics', ''))
            ocean_scores = persona.get('ocean_scores', {})
            
            return PersonaResponse(
                persona_id=str(persona.get('id', '?')),
                will_click=will_click,
                reasoning=reasoning,
                demographics=demographics,
                ocean_scores=ocean_scores
            )
    
    async def predict_async(
        self,
        ad_text: Optional[str] = None,
        image_url: Optional[str] = None,
        population_size: int = 100,
        ad_platform: str = 'facebook'
    ) -> CTRPredictionResult:
        """Predict CTR for an advertisement (async).
        
        Args:
            ad_text: Text of the advertisement (for text ads)
            image_url: URL of the advertisement image (for image ads)
            population_size: Number of personas to evaluate
            ad_platform: Platform where ad is shown
            
        Returns:
            CTRPredictionResult object
        """
        # Validate input
        if not ad_text and not image_url:
            raise ValueError("Either ad_text or image_url must be provided")
        
        ad_type = "IMAGE AD" if image_url else "TEXT AD"
        ad_content = image_url if image_url else ad_text
        
        print("\n" + "="*80)
        print(f"CTR PREDICTION - {ad_type}")
        print("="*80)
        print(f"Ad Platform: {ad_platform}")
        print(f"Ad Content: {ad_content}")
        print(f"Population Size: {population_size:,}")
        print(f"Persona Version: {self.persona_version}")
        print(f"Persona Strategy: {self.persona_strategy}")
        print("="*80 + "\n")
        
        # Load personas
        personas = self._load_personas(population_size)
        
        # Get responses from each persona
        print(f"\nCollecting responses from {len(personas):,} personas...")
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        tasks = [
            self._get_persona_response_async(persona, ad_text, image_url, ad_platform, semaphore)
            for persona in personas
        ]
        
        persona_responses = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Processing personas"):
            response = await coro
            persona_responses.append(response)
        
        # Calculate CTR
        total_clicks = sum(1 for r in persona_responses if r.will_click)
        error_count = sum(1 for r in persona_responses if 'API Error' in r.reasoning)
        ctr = total_clicks / len(persona_responses) if persona_responses else 0.0
        
        print(f"\n✓ Collected {len(persona_responses):,} responses")
        print(f"  Predicted Clicks: {total_clicks}")
        print(f"  Predicted CTR: {ctr:.2%}")
        if error_count > 0:
            print(f"  ⚠️  API Errors: {error_count} ({error_count/len(persona_responses)*100:.1f}% of requests failed)")
            print(f"     These responses were excluded from analysis.")
        
        # Generate final analysis
        print(f"\nGenerating final analysis...")
        final_analysis = await generate_final_analysis(
            ad_content, ad_platform, persona_responses, ctr
        )
        
        print("✓ Analysis complete\n")
        
        return CTRPredictionResult(
            ctr=ctr,
            total_personas=len(persona_responses),
            total_clicks=total_clicks,
            persona_responses=persona_responses,
            final_analysis=final_analysis,
            provider_used="gpt-4o-mini + deepseek-chat",
            model_used="gpt-4o-mini (decisions), deepseek-chat (analysis)",
            ad_platform=ad_platform
        )
    
    def predict(
        self,
        ad_text: Optional[str] = None,
        image_url: Optional[str] = None,
        population_size: int = 100,
        ad_platform: str = 'facebook'
    ) -> CTRPredictionResult:
        """Predict CTR for an advertisement (synchronous wrapper).
        
        Args:
            ad_text: Text of the advertisement (for text ads)
            image_url: URL of the advertisement image (for image ads)
            population_size: Number of personas to evaluate
            ad_platform: Platform where ad is shown
            
        Returns:
            CTRPredictionResult object
        """
        return asyncio.run(self.predict_async(ad_text, image_url, population_size, ad_platform))
