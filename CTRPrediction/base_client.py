"""Base client interface for LLM click prediction."""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _compact_profile_str(idx: int, p: Dict[str, Any]) -> str:
    """Serialize a persona into a compact representation.

    The format includes: index, demographics, personality scores, and persona description.

    Args:
        idx: Row index of the persona.
        p: Persona dictionary with demographics, personality, and description.

    Returns:
        String representation of the persona.
    """
    # Extract demographics
    demographics = p.get("demographics", {})
    age = demographics.get("age", "N/A")
    gender = demographics.get("gender", "N/A")
    state = demographics.get("state", "N/A")
    race = demographics.get("race", "N/A")
    education = demographics.get("educational_attainment", "N/A")
    occupation = demographics.get("occupation", "N/A")
    
    # Extract personality scores
    personality = p.get("personality", {})
    scores = personality.get("scores", {})
    openness = scores.get("openness", "N/A")
    conscientiousness = scores.get("conscientiousness", "N/A")
    extraversion = scores.get("extraversion", "N/A")
    agreeableness = scores.get("agreeableness", "N/A")
    neuroticism = scores.get("neuroticism", "N/A")
    
    # Get persona description
    description = p.get("description", "N/A")
    
    return (
        f"[Persona {idx}]\n"
        f"Demographics: Age={age}, Gender={gender}, State={state}, Race={race}, Education={education}, Occupation={occupation}\n"
        f"Personality (Big Five): Openness={openness}, Conscientiousness={conscientiousness}, Extraversion={extraversion}, Agreeableness={agreeableness}, Neuroticism={neuroticism}\n"
        f"Description: {description}\n"
    )


def _build_prompt(ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> str:
    """Compose the LLM prompt for batched binary click decisions using persona profiles.

    The prompt includes the ad text, platform context, detailed persona profiles with
    demographics, Big Five personality traits, and persona descriptions, along with strict
    instructions to return a JSON array of 0/1 integers only.

    Args:
        ad_text: The advertisement copy to evaluate.
        profiles: List of persona profiles with demographics, personality, and descriptions.
        ad_platform: Platform where the ad is shown (facebook, tiktok, amazon).

    Returns:
        A single prompt string.
    """
    # Platform-specific context
    platform_contexts = {
        "facebook": "This ad appears in users' Facebook news feed while they browse social content.",
        "tiktok": "This ad appears between TikTok videos as users scroll through short-form content.",
        "amazon": "This ad appears on Amazon while users are shopping or browsing products."
    }
    
    platform_context = platform_contexts.get(ad_platform.lower(), platform_contexts["facebook"])
    
    header = (
        "Task: You are evaluating whether people would click on an advertisement based on their detailed personas.\n"
        f"Platform Context: {platform_context}\n\n"
        "Advertisement:\n"
    )
    
    rules = (
        "\n\nInstructions:\n"
        "- For each persona below, decide if that person would click the ad based on their demographics, personality traits (Big Five), and persona description.\n"
        "- Consider how the ad content aligns with their interests, values, needs, and behavioral tendencies.\n"
        "- Output ONLY a simple JSON list of 0/1 integers, no commentary or explanation.\n"
        "- Format: [0,1,0,1] (compact, NO SPACES between elements)\n"
        "- NOT this: [0, 1, 0, 1] (no spaces after commas)\n"
        "- NOT objects: [{'0':0}] or nested structures\n"
        "- The array length must equal the number of personas.\n"
        "- 1 means 'would click', 0 means 'would not click'.\n"
        "- Stop after closing bracket ']'\n"
    )
    
    personas_header = "\n\nPersonas to evaluate:\n" + "="*80 + "\n\n"
    
    lines = [
        _compact_profile_str(i, p) for i, p in enumerate(profiles)
    ]
    
    return header + ad_text + rules + personas_header + "\n".join(lines)


def _fix_truncated_response(s: str, expected_length: int) -> str:
    """Fix truncated model responses by trimming to exact expected length.
    
    Open source models often pad responses to max_tokens. This function:
    1. Calculates the expected compact JSON format length: [0,1,0,1]
    2. Trims the response to that exact length
    3. Ensures proper JSON format with closing bracket
    
    Args:
        s: Raw response text that may be truncated/padded
        expected_length: Number of elements expected in the array
        
    Returns:
        Fixed response string in compact JSON format
    """
    s = s.strip()
    
    # Expected format: [0,1,0,1] = 1 + (expected_length * 2 - 1) + 1
    # = opening bracket + "0,1,0,1" pattern + closing bracket
    target_chars = 1 + (expected_length * 2 - 1) + 1
    
    if len(s) > target_chars:
        # Truncate padded response to exact length
        s = s[:target_chars-1] + ']'
    elif len(s) == target_chars - 1 and not s.endswith(']'):
        # Add missing closing bracket
        s = s + ']'
    
    return s


def _try_parse_json_array(s: str, expected_length: Optional[int] = None) -> Optional[List[int]]:
    """Extract a JSON array of integers from a model response string.

    Tries direct JSON parsing, then a tolerant regex extraction if wrapped in
    extra text. For open source models, applies truncation fix if expected_length provided.

    Args:
        s: Raw response text.
        expected_length: Expected number of elements (for truncation fix)

    Returns:
        List of integers on success, otherwise ``None``.
    """
    # Apply truncation fix for open source models that pad to max_tokens
    if expected_length is not None:
        s = _fix_truncated_response(s, expected_length)
    
    try:
        data = json.loads(s)
        if isinstance(data, list):
            # Handle simple list format: [0, 1, 1, 0] (preferred)
            if all(isinstance(x, int) for x in data):
                return data
            # Handle object format: [{"0":0}, {"0":1}] (backward compatibility)
            elif all(isinstance(x, dict) and len(x) == 1 for x in data):
                result = []
                for item in data:
                    value = list(item.values())[0]
                    if isinstance(value, int):
                        result.append(value)
                    else:
                        return None
                return result
    except Exception:
        pass
    m = re.search(r"\[(?:\s*\d\s*,?\s*)+\]", s)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list) and all(isinstance(x, int) for x in data):
                return data
        except Exception:
            return None
    return None


def _print_fallback(msg: str) -> None:
    """Print a concise notice when falling back to the mock model.

    Args:
        msg: Short human-readable reason, e.g., provider/step that failed.
    """
    try:
        print(f"[Mock Fallback] {msg}")
    except Exception:
        pass


def _sigmoid(x: float) -> float:
    """Logistic sigmoid function with a safe fallback.

    Args:
        x: Input value.

    Returns:
        Sigmoid(x) in (0, 1). Returns 0.5 if math operations fail.
    """
    try:
        import math
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def _mock_click_prob(ad_text: str, p: Dict[str, Any]) -> float:
    """Heuristic click propensity for use in mock predictions with persona profiles.

    Combines ad keyword matches with persona demographics, personality traits,
    and description to compute a probability in [0, 1], passed through a sigmoid.

    Args:
        ad_text: Advertisement text.
        p: Persona dictionary with demographics, personality, and description.

    Returns:
        A probability between 0 and 1.
    """
    text = (ad_text or "").lower()
    
    # Extract demographics
    demographics = p.get("demographics", {})
    age = int(demographics.get("age", 40))
    gender = (demographics.get("gender") or "").lower()
    occupation = (demographics.get("occupation") or "").lower()
    education = (demographics.get("educational_attainment") or "").lower()
    
    # Extract personality scores (Big Five: 1-10 scale typically)
    personality = p.get("personality", {})
    scores = personality.get("scores", {})
    openness = int(scores.get("openness", 5))
    conscientiousness = int(scores.get("conscientiousness", 5))
    extraversion = int(scores.get("extraversion", 5))
    agreeableness = int(scores.get("agreeableness", 5))
    neuroticism = int(scores.get("neuroticism", 5))
    
    # Get persona description for additional context
    description = (p.get("description") or "").lower()

    z = -0.5

    # Keyword-based scoring with personality modifiers
    if any(k in text for k in ["credit", "loan", "debt", "card", "financial"]):
        # Higher conscientiousness and lower neuroticism = more interested in financial products
        z += 0.3 + (conscientiousness - 5) * 0.1 - (neuroticism - 5) * 0.05
    
    if any(k in text for k in ["health", "medical", "insurance", "clinic", "wellness"]):
        # Higher conscientiousness and neuroticism = more health-conscious
        z += 0.4 + (conscientiousness - 5) * 0.1 + (neuroticism - 5) * 0.08
    
    if any(k in text for k in ["travel", "flight", "hotel", "vacation", "adventure"]):
        # Higher openness and extraversion = love for travel
        z += 0.5 + (openness - 5) * 0.15 + (extraversion - 5) * 0.1
    
    if any(k in text for k in ["education", "course", "degree", "learn", "study"]):
        # Higher openness = interest in learning
        z += 0.5 + (openness - 5) * 0.15
        if "student" in occupation or "teacher" in occupation or age < 30:
            z += 0.4
    
    if any(k in text for k in ["retire", "pension", "senior"]):
        z += 0.7 if age >= 60 else -0.3
    
    if any(k in text for k in ["tech", "gadget", "software", "app", "gaming", "ai"]):
        # Higher openness and younger age = interest in tech
        z += 0.4 + (openness - 5) * 0.1
        if age < 40 or "engineer" in occupation or "tech" in occupation:
            z += 0.3
    
    if any(k in text for k in ["home", "furniture", "kitchen", "decor"]):
        # Higher conscientiousness = interest in home improvement
        z += 0.3 + (conscientiousness - 5) * 0.08
    
    if any(k in text for k in ["fashion", "beauty", "clothing", "style"]):
        # Higher openness and extraversion = fashion interest
        z += 0.2 + (openness - 5) * 0.08 + (extraversion - 5) * 0.08
        if gender == "female":
            z += 0.2
    
    if any(k in text for k in ["sports", "fitness", "gym", "exercise"]):
        # Higher extraversion and conscientiousness = fitness interest
        z += 0.2 + (extraversion - 5) * 0.08 + (conscientiousness - 5) * 0.06
        if age < 45:
            z += 0.2
    
    if any(k in text for k in ["social", "community", "event", "party"]):
        # Higher extraversion and agreeableness = social events
        z += 0.3 + (extraversion - 5) * 0.15 + (agreeableness - 5) * 0.08
    
    if any(k in text for k in ["art", "music", "creative", "design"]):
        # Higher openness = interest in arts
        z += 0.4 + (openness - 5) * 0.15
    
    # Age-based adjustments
    if age < 25:
        z += 0.1  # Younger people more likely to click in general
    elif age > 65:
        z -= 0.15  # Older people more cautious
    
    # Education level boost for complex/professional products
    if any(k in text for k in ["professional", "premium", "executive", "luxury"]):
        if "bachelor" in education or "master" in education or "professional degree" in education:
            z += 0.3

    return max(0.0, min(1.0, _sigmoid(z)))


def _mock_predict(ad_text: str, profiles: List[Dict[str, Any]]) -> List[int]:
    """Generate binary click predictions using the mock heuristic.

    Uses a fixed RNG seed for reproducibility across runs.

    Args:
        ad_text: Advertisement text.
        profiles: List of profiles to score.

    Returns:
        List of 0/1 integers aligned with ``profiles``.
    """
    import random
    rnd = random.Random(1337)
    clicks = []
    for p in profiles:
        prob = _mock_click_prob(ad_text, p)
        clicks.append(1 if rnd.random() < prob else 0)
    return clicks


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, api_key: str = None):
        """Initialize base client.
        
        Args:
            model: Model name to use.
            api_key: API key override.
        """
        self.model = model
        self.api_key = api_key
        self.provider_name = "base"
        self.env_key_name = "API_KEY"
        self.used_fallback = False  # Track if mock fallback was used
    
    def has_api_key(self) -> bool:
        """Check if API key is available."""
        key = self.api_key or os.getenv(self.env_key_name)
        return bool(key)
    
    def _build_prompt(self, ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> str:
        """Build prompt for the profiles."""
        return _build_prompt(ad_text, profiles, ad_platform)
    
    def _fallback_to_mock(self, ad_text: str, chunk: List[Dict[str, Any]], reason: str) -> List[int]:
        """Fallback to mock prediction with logging."""
        self.used_fallback = True  # Mark that fallback was used
        _print_fallback(f"{reason}; using mock for this chunk.")
        return _mock_predict(ad_text, chunk)
    
    def _parse_and_validate_response(self, content: str, chunk: List[Dict[str, Any]], ad_text: str, provider_name: str) -> List[int]:
        """Parse and validate LLM response."""
        # Pass expected length for truncation fix on open source models
        expected_length = len(chunk)
        arr = _try_parse_json_array(content or "", expected_length)
        if arr is None:
            self.used_fallback = True  # Mark that fallback was used
            # Debug: Print raw response when parsing fails
            print(f"[DEBUG] Parse failed - Content length: {len(content) if content else 0}")
            print(f"[DEBUG] Content preview: {repr(content if content else 'None')}")
            _print_fallback(f"{provider_name} output parse failed; using mock for this chunk.")
            arr = _mock_predict(ad_text, chunk)
        
        # Pad or trim to correct length
        if len(arr) != len(chunk):
            if len(arr) < len(chunk):
                arr = arr + [0] * (len(chunk) - len(arr))
            else:
                arr = arr[: len(chunk)]
        
        return [1 if int(x) else 0 for x in arr]
    
    @abstractmethod
    def predict_chunk(self, ad_text: str, chunk: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles synchronously.
        
        Args:
            ad_text: Advertisement text.
            chunk: List of profile dictionaries.
            ad_platform: Platform where ad is shown.
            
        Returns:
            List of 0/1 click predictions.
        """
        pass
    
    @abstractmethod
    async def predict_chunk_async(self, ad_text: str, chunk: List[Dict[str, Any]], ad_platform: str = "facebook") -> List[int]:
        """Predict clicks for a chunk of profiles asynchronously.
        
        Args:
            ad_text: Advertisement text.
            chunk: List of profile dictionaries.
            ad_platform: Platform where ad is shown.
            
        Returns:
            List of 0/1 click predictions.
        """
        pass