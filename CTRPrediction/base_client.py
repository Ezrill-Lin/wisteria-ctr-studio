"""Base client interface for LLM click prediction."""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _compact_profile_str(idx: int, p: Dict[str, Any]) -> str:
    """Serialize a profile into a compact pipe-delimited line.

    The columns are: index|gender|age|region|occupation|salary|liability|married|health|illness.

    Args:
        idx: Row index of the profile.
        p: Profile dictionary with identity attributes.

    Returns:
        Pipe-delimited string representation of the profile.
    """
    illness = p.get("illness", "-") if p.get("health_status") else "-"
    married = 1 if p.get("is_married") else 0
    health = 1 if p.get("health_status") else 0
    return (
        f"{idx}|{p.get('gender')}|{p.get('age')}|{p.get('region')}|"
        f"{p.get('occupation')}|{p.get('annual_salary')}|{p.get('liability_status')}|"
        f"{married}|{health}|{illness}"
    )


def _build_prompt(ad_text: str, profiles: List[Dict[str, Any]], ad_platform: str = "facebook") -> str:
    """Compose the LLM prompt for batched binary click decisions.

    The prompt includes the ad text, platform context, concise profile rows, and strict
    instructions to return a JSON array of 0/1 integers only.

    Args:
        ad_text: The advertisement copy to evaluate.
        profiles: List of identity profiles.
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
        "Task: For each profile below, decide if that person would click the ad.\n"
        f"Platform Context: {platform_context}\n"
        "Ad: "
    )
    rules = (
        "\n\nRules:\n"
        "- Output ONLY a JSON array of 0/1 integers, no commentary.\n"
        "- The array length must equal the number of profiles.\n"
        "- 1 means 'would click', 0 means 'would not click'.\n"
        "- Consider the platform context when making decisions.\n"
    )
    cols = (
        "\n\nProfiles (index|gender|age|region|occupation|salary|liability|married|health|illness):\n"
    )
    lines = [
        _compact_profile_str(i, p) for i, p in enumerate(profiles)
    ]
    return header + ad_text + rules + cols + "\n".join(lines)


def _try_parse_json_array(s: str) -> Optional[List[int]]:
    """Extract a JSON array of integers from a model response string.

    Tries direct JSON parsing, then a tolerant regex extraction if wrapped in
    extra text.

    Args:
        s: Raw response text.

    Returns:
        List of integers on success, otherwise ``None``.
    """
    try:
        data = json.loads(s)
        if isinstance(data, list) and all(isinstance(x, int) for x in data):
            return data
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
    """Heuristic click propensity for use in mock predictions.

    Combines ad keyword matches with profile attributes to compute a
    probability in [0, 1], passed through a sigmoid.

    Args:
        ad_text: Advertisement text.
        p: Profile dictionary.

    Returns:
        A probability between 0 and 1.
    """
    text = (ad_text or "").lower()
    age = int(p.get("age", 40))
    salary = float(p.get("annual_salary", 50000.0))
    liab = float(p.get("liability_status", 10000.0))
    married = 1 if p.get("is_married") else 0
    health = 1 if p.get("health_status") else 0
    occ = (p.get("occupation") or "").lower()
    gender = (p.get("gender") or "").lower()

    z = -0.5

    if any(k in text for k in ["credit", "loan", "debt", "card"]):
        z += min(liab / 100000.0, 2.0)
    if any(k in text for k in ["health", "medical", "insurance", "clinic"]):
        z += 0.8 * health
    if any(k in text for k in ["travel", "flight", "hotel", "vacation"]):
        z += min((salary - 40000.0) / 60000.0, 1.0)
    if any(k in text for k in ["education", "course", "degree", "learn"]):
        z += 0.6 if ("student" in occ or "teacher" in occ or age < 30) else 0.1
    if any(k in text for k in ["retire", "pension"]):
        z += 0.7 if age >= 60 else -0.3
    if any(k in text for k in ["tech", "gadget", "software", "app", "gaming", "ai"]):
        z += 0.4 if (age < 40 or "engineer" in occ) else 0.1
    if any(k in text for k in ["home", "mortgage", "furniture", "kitchen"]):
        z += 0.3 * married
    if any(k in text for k in ["fashion", "beauty", "clothing"]):
        z += 0.2 if gender == "female" else 0.05
    if any(k in text for k in ["sports", "fitness", "gym"]):
        z += 0.2 if age < 45 else 0.05

    z += (salary - 60000.0) / 120000.0
    z -= liab / 300000.0
    z += (0.15 if married else 0.0)

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
        arr = _try_parse_json_array(content or "")
        if arr is None:
            self.used_fallback = True  # Mark that fallback was used
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