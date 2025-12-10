"""
CTR Predictor Module

Unified predictor for both text and image-based advertisements using Gemini.
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
        concurrent_requests: int = 10,
        realistic_mode: bool = True,
        decision_model: str = 'gemini-2.5-flash-lite'
    ):
        """Initialize CTR predictor.
        
        Args:
            persona_version: Version of personas to use ('v1' or 'v2')
            persona_strategy: Strategy for persona generation ('random', 'wpp', 'ipip')
            concurrent_requests: Number of concurrent API calls
            realistic_mode: If True, uses enhanced real-world browsing context (recommended)
            decision_model: Model for click decisions ('gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-flash')
        """
        self.persona_version = persona_version
        self.persona_strategy = persona_strategy
        self.concurrent_requests = concurrent_requests
        self.realistic_mode = realistic_mode
        self.decision_model = decision_model
        
        # Validate API key
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        if not decision_model.startswith('gemini'):
            raise ValueError(f"Unsupported model: {decision_model}. Only Gemini models are supported.")
        
        print(f"✓ Initialized CTR Predictor")
        print(f"  Unified Model: {decision_model} (decisions & analysis)")
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
            else:
                # Text ad with realistic mode option
                user_prompt = create_persona_user_prompt(ad_text, ad_platform, realistic_mode=self.realistic_mode)
            
            # Call LLM for click decisions with retry logic
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    if self.decision_model.startswith('gemini'):
                        # Use Gemini API (new google-genai SDK pattern)
                        from google import genai
                        
                        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                        
                        # Combine system and user messages for Gemini
                        full_prompt = f"{system_message}\n\n{user_prompt}"
                        
                        # Generate response (async)
                        response = await client.aio.models.generate_content(
                            model=self.decision_model,
                            contents=full_prompt,
                            config=genai.types.GenerateContentConfig(
                                temperature=0.3,
                                top_p=0.9,
                                max_output_tokens=8000  # Match SiliconSampling to prevent truncation
                            )
                        )
                        
                        # Get text from response
                        content = response.text
                        if not content:
                            # Check finish reason like SiliconSampling does
                            finish_reason = response.candidates[0].finish_reason if response.candidates else 'Unknown'
                            raise ValueError(f"Gemini returned empty text. Finish reason: {finish_reason}")
                        
                        # Strip markdown code blocks if present (Gemini wraps JSON in ```json ... ```)
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content[7:]  # Remove ```json
                        elif content.startswith('```'):
                            content = content[3:]  # Remove ```
                        if content.endswith('```'):
                            content = content[:-3]  # Remove closing ```
                        content = content.strip()
                        
                        # Parse JSON response
                        will_click, reasoning = parse_json_response(content)
                        break  # Success, exit retry loop
                    
                except Exception as e:
                    error_type = type(e).__name__
                    
                    # Retry on rate limit or timeout errors
                    if attempt < max_retries - 1 and ('rate' in str(e).lower() or 'timeout' in str(e).lower() or '429' in str(e) or '503' in str(e)):
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
            provider_used=f"{self.decision_model}",
            model_used=f"{self.decision_model} (decisions & analysis)",
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
