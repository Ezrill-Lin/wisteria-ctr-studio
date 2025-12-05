"""
Client utilities for API interactions

This module handles:
- API client initialization (OpenAI, DeepSeek, Google Gemini)
- Async response generation
- Response collection with concurrency control
"""

import os
import json
import asyncio
from google import genai
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from agent_utils import save_responses_batch, create_prompt, detect_api_provider


async def get_response_async(client, persona, questions, model="gpt-4o-mini", max_retries=3, use_json_mode=True, api_provider="openai"):
    """
    Async call to get LLM response for a persona
    
    Args:
        client: AsyncOpenAI client or genai model (for Gemini)
        persona: Persona dictionary
        questions: Questions to answer
        model: Model to use (gpt-4o-mini, deepseek-chat, gemini-2.5-flash, etc.)
        max_retries: Number of retry attempts
        use_json_mode: Whether to use JSON response format (not all models support this)
        api_provider: API provider ('openai', 'deepseek', or 'gemini')
    
    Returns:
        Response dictionary or None if failed
    """
    # Get system message and user prompt
    system_message, user_prompt = create_prompt(persona, questions)
    
    # Gemini API uses different calling pattern
    if api_provider == "gemini":
        return await get_gemini_response(client, system_message, user_prompt, max_retries)
    
    # OpenAI/DeepSeek pattern
    for attempt in range(max_retries):
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2000
            }
            
            # Only add response_format if supported
            if use_json_mode:
                params["response_format"] = {"type": "json_object"}
            
            response = await client.chat.completions.create(**params)
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
            
        except Exception as e:
            # Print detailed error on first attempt to help debug issues
            if attempt == 0:
                error_msg = str(e).lower()
                print(f"\n❌ {api_provider.upper()} API error: {type(e).__name__}: {e}")
                
                # Check if it's a model-related error
                if "model" in error_msg or "not found" in error_msg or "does not exist" in error_msg:
                    print(f"   ❌ Model '{model}' may not exist or you may not have access.")
                    print(f"   Stopping execution to prevent wasting API calls.\n")
                    # Re-raise to stop execution
                    raise ValueError(f"Invalid model name: {model}") from e
                    
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
    
    return None


async def get_gemini_response(client, system_message, user_prompt, max_retries=3):
    """
    Get response from Gemini API
    
    Args:
        client: Gemini client instance
        system_message: System instructions
        user_prompt: User prompt
        max_retries: Number of retry attempts
    
    Returns:
        Response dictionary or None if failed
    """
    for attempt in range(max_retries):
        try:
            # Gemini combines system message into the prompt
            full_prompt = f"{system_message}\n\n{user_prompt}"
            
            # Generate response (async)
            response = await client.aio.models.generate_content(
                model=client.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=8000,  # Increased to prevent truncation
                )
            )
            
            # Get text from response - use .text attribute directly
            text = response.text
            if not text:
                print(f"Gemini returned empty text. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None
            
            # Strip markdown code blocks if present (Gemini wraps JSON in ```json ... ```)
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]  # Remove ```json
            elif text.startswith('```'):
                text = text[3:]  # Remove ```
            if text.endswith('```'):
                text = text[:-3]  # Remove closing ```
            text = text.strip()
            
            # Parse JSON from response
            result = json.loads(text)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Gemini JSON decode error: {e}")
            print(f"Response text: {text[:200] if 'text' in locals() else 'No text'}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
            
        except Exception as e:
            # Print detailed error on first attempt to help debug issues
            if attempt == 0:
                error_msg = str(e).lower()
                print(f"\n❌ Gemini API error: {type(e).__name__}: {e}")
                
                # Check if it's a model-related error
                if "model" in error_msg or "not found" in error_msg or "does not exist" in error_msg:
                    print(f"   ❌ Model '{client.model}' may not exist or you may not have access.")
                    print(f"   Stopping execution to prevent wasting API calls.\n")
                    # Re-raise to stop execution
                    raise ValueError(f"Invalid model name: {client.model}") from e
                    
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None
    
    return None


async def collect_responses(personas, questions, strategy, model="gpt-4o-mini", api_provider="openai",
                           concurrent_requests=20, batch_save_size=50, response_version='v1'):
    """
    Collect responses from all personas using async processing
    
    Args:
        personas: List of persona dictionaries
        questions: Questions to answer
        strategy: Strategy name for output file
        model: Model to use (e.g., 'gemini-2.5-flash', 'deepseek-chat', 'gpt-4o')
        concurrent_requests: Number of concurrent API calls
        batch_save_size: Save progress every N responses
        response_version: Response version folder ('v1' or 'v2')
    
    Returns:
        Path to responses file
    """
    # Initialize client based on provider
    if api_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables!\n"
                "Set it with: $env:GEMINI_API_KEY='your-key-here' (PowerShell)"
            )
        client = genai.Client(api_key=api_key)
        client.model = model  # Store model name for later use
        use_json_mode = False  # Gemini handles JSON differently
        
    elif api_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
        use_json_mode = False  # DeepSeek doesn't support response_format
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment variables!\n"
                "Set it with: $env:DEEPSEEK_API_KEY='your-key-here' (PowerShell)"
            )
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")
        use_json_mode = True
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables!\n"
                "Set it with: $env:OPENAI_API_KEY='your-key-here' (PowerShell)"
            )
        client = AsyncOpenAI(api_key=api_key)
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    failed_count = 0
    model_error_detected = False
    
    async def process_one(persona):
        nonlocal failed_count, model_error_detected
        
        async with semaphore:
            # Stop processing if model error was detected
            if model_error_detected:
                return None
                
            response = await get_response_async(
                client, persona, questions, 
                model=model, use_json_mode=use_json_mode, api_provider=api_provider
            )
            if response is None:
                failed_count += 1
                response = 'FAILED'
            
            return {
                'persona_id': persona['id'],
                'responses': response
            }
    
    # Create all tasks
    tasks = [process_one(p) for p in personas]
    
    # Run with progress bar
    print("\nCollecting responses...")
    results = []
    output_file = Path(f"responses_{response_version}/{strategy}/{api_provider}/responses_{response_version}_{strategy}_{model}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Responses", ncols=100):
        result = await coro
        results.append(result)
        
        # Save batch
        if len(results) >= batch_save_size:
            save_responses_batch(results, output_file, append=True)
            results = []
    
    # Save remaining
    if results:
        save_responses_batch(results, output_file, append=True)
    
    if failed_count > 0:
        print(f"\n⚠ Warning: {failed_count}/{len(personas)} responses failed")
    
    return output_file
