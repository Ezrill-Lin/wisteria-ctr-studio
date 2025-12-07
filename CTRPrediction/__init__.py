"""CTRPrediction (Click-Through Rate Prediction) Package

This package provides specialized tools for CTR (Click-Through Rate) prediction
using synthetic populations and LLM-based persona reasoning.

Main Components:
- CTRPredictor: Unified predictor for both text and image advertisements
- PersonaResponse: Individual persona's click decision with reasoning
- CTRPredictionResult: Complete prediction results with CTR, responses, and analysis

Features:
- Text & Image Support: Handle both text and image ads using gpt-4o-mini with vision capabilities
- Persona-based prediction: Uses SiliconSampling v1/v2 personas with full psychological profiles
- Individual reasoning: Each persona provides click decision + 1-3 sentence reasoning
- Final analysis: DeepSeek-powered insights about ad effectiveness and recommendations
- Async processing: Concurrent API calls with semaphore control
"""

from .ctr_predictor import CTRPredictor, PersonaResponse, CTRPredictionResult

__version__ = "3.1.0"
__all__ = [
    "CTRPredictor",
    "CTRPredictionResult",
    "PersonaResponse",
]