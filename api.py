"""
Wisteria CTR Studio - FastAPI Service

This module provides a REST API interface for CTR prediction using SiliconSampling personas.

Usage:
    # Start the development server
    python api.py
    
    # Or use uvicorn directly
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload
    
    # API docs available at: http://localhost:8080/docs
"""

import asyncio
import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from CTRPrediction import CTRPredictor, CTRPredictionResult
from CTRPrediction.utils import encode_image_to_data_url


# Pydantic models for request/response
class CTRTextRequest(BaseModel):
    """Request model for text ad CTR prediction."""
    ad_text: str = Field(
        description="Advertisement text content",
        min_length=1,
        example="Special 0% APR credit card offer for travel rewards"
    )
    population_size: int = Field(
        default=1000,
        description="Number of personas to evaluate",
        ge=1,
        le=1000000,
        example=1000
    )
    ad_platform: str = Field(
        default="facebook",
        description="Platform where ad is shown",
        example="facebook"
    )
    persona_version: str = Field(
        default="v2",
        description="Persona version (v1 or v2)",
        pattern="^(v1|v2)$",
        example="v2"
    )
    persona_strategy: str = Field(
        default="random",
        description="Persona generation strategy (random, wpp, ipip)",
        pattern="^(random|wpp|ipip)$",
        example="random"
    )
    concurrent_requests: int = Field(
        default=10,
        description="Number of concurrent API requests",
        ge=1,
        le=50,
        example=10
    )
    include_persona_details: bool = Field(
        default=False,
        description="Include individual persona responses in output",
        example=False
    )
    realistic_mode: bool = Field(
        default=True,
        description="Use enhanced real-world browsing context (recommended for accurate CTR)",
        example=True
    )
    decision_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Model for click decisions and analysis (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-1.5-flash)",
        example="gemini-2.5-flash-lite"
    )

    class Config:
        schema_extra = {
            "example": {
                "ad_text": "Special 0% APR credit card offer for travel rewards",
                "population_size": 100,
                "ad_platform": "facebook",
                "persona_version": "v2",
                "persona_strategy": "random",
                "concurrent_requests": 20,
                "include_persona_details": False
            }
        }


class CTRImageRequest(BaseModel):
    """Request model for image ad CTR prediction."""
    image_url: str = Field(
        description="Advertisement image URL (http/https or data URL)",
        min_length=1,
        example="https://example.com/ad-image.jpg"
    )
    population_size: int = Field(
        default=100,
        description="Number of personas to evaluate",
        ge=1,
        le=10000,
        example=100
    )
    ad_platform: str = Field(
        default="facebook",
        description="Platform where ad is shown",
        example="facebook"
    )
    persona_version: str = Field(
        default="v2",
        description="Persona version (v1 or v2)",
        pattern="^(v1|v2)$",
        example="v2"
    )
    persona_strategy: str = Field(
        default="random",
        description="Persona generation strategy (random, wpp, ipip)",
        pattern="^(random|wpp|ipip)$",
        example="random"
    )
    concurrent_requests: int = Field(
        default=20,
        description="Number of concurrent API requests",
        ge=1,
        le=50,
        example=20
    )
    include_persona_details: bool = Field(
        default=False,
        description="Include individual persona responses in output",
        example=False
    )
    realistic_mode: bool = Field(
        default=True,
        description="Use enhanced real-world browsing context (recommended for accurate CTR)",
        example=True
    )

    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://example.com/ad-image.jpg",
                "population_size": 100,
                "ad_platform": "facebook",
                "persona_version": "v2",
                "persona_strategy": "random",
                "concurrent_requests": 20,
                "include_persona_details": False
            }
        }


class PersonaResponseModel(BaseModel):
    """Individual persona response."""
    persona_id: str
    will_click: bool
    reasoning: str
    demographics: str = ""
    age: Optional[str] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    education: Optional[str] = None
    location: Optional[str] = None


class CTRResponse(BaseModel):
    """Response model for CTR prediction."""
    success: bool = True
    ctr: float
    total_clicks: int
    total_personas: int
    runtime_seconds: float
    provider_used: str
    model_used: str
    ad_platform: str
    persona_version: str
    persona_strategy: str
    timestamp: str
    final_analysis: str
    persona_responses: Optional[List[PersonaResponseModel]] = None

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "ctr": 0.247,
                "total_clicks": 247,
                "total_personas": 1000,
                "runtime_seconds": 45.6,
                "provider_used": "google",
                "model_used": "gemini-2.5-flash",
                "ad_platform": "facebook",
                "persona_version": "v2",
                "persona_strategy": "random",
                "timestamp": "2025-12-06T10:30:00Z",
                "final_analysis": "This ad has a predicted CTR of 24.7%, which is above average...",
                "persona_responses": None
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    model: str
    available_persona_versions: List[str]
    available_persona_strategies: List[str]
    available_platforms: List[str]


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Wisteria CTR Studio API",
    description="REST API for CTR prediction using SiliconSampling synthetic personas and LLM-based behavioral simulation. Supports both text and image advertisements.",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
AVAILABLE_PERSONA_VERSIONS = ["v1", "v2"]
AVAILABLE_PERSONA_STRATEGIES = ["random", "wpp", "ipip"]
AVAILABLE_PLATFORMS = ["facebook", "tiktok", "amazon", "instagram", "youtube"]


def parse_demographics_string(demographics_str: str) -> dict:
    """Parse demographics string into structured fields.
    
    Args:
        demographics_str: Demographics string like "69-year-old, female, working as..."
        
    Returns:
        Dictionary with age, gender, occupation, education, location
    """
    import re
    
    result = {
        "age": None,
        "gender": None,
        "occupation": None,
        "education": None,
        "location": None
    }
    
    if not demographics_str:
        return result
    
    # Extract age: "XX-year-old"
    age_match = re.search(r'(\d+)-year-old', demographics_str)
    if age_match:
        result["age"] = age_match.group(1)
    
    # Extract gender: "male" or "female"
    gender_match = re.search(r'\b(male|female)\b', demographics_str, re.IGNORECASE)
    if gender_match:
        result["gender"] = gender_match.group(1).capitalize()
    
    # Extract occupation: "working as XXX"
    occupation_match = re.search(r'working as ([^,]+)', demographics_str)
    if occupation_match:
        occ = occupation_match.group(1).strip()
        # Clean up common patterns
        if "N/A" in occ or "less than 16" in occ or "NILF" in occ:
            result["occupation"] = "Not in Labor Force"
        else:
            result["occupation"] = occ
    
    # Extract education: "with XXX"
    education_match = re.search(r'with ([^,]+?)(?:,\s*living|$)', demographics_str)
    if education_match:
        result["education"] = education_match.group(1).strip()
    
    # Extract location: "living in XXX"
    location_match = re.search(r'living in ([^,]+)', demographics_str)
    if location_match:
        result["location"] = location_match.group(1).strip()
    
    return result


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Wisteria CTR Studio API",
        "version": "3.1.0",
        "description": "REST API for CTR prediction using SiliconSampling personas",
        "models": {
            "unified_model": "gemini-2.5-flash-lite (click decisions & analysis, supports text and images)"
        },
        "endpoints": {
            "health": "/health",
            "predict_text": "/predict/text",
            "predict_image": "/predict/image",
            "models": "/models",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="3.1.0",
        model="gemini-2.5-flash-lite (unified for decisions & analysis)",
        available_persona_versions=AVAILABLE_PERSONA_VERSIONS,
        available_persona_strategies=AVAILABLE_PERSONA_STRATEGIES,
        available_platforms=AVAILABLE_PLATFORMS
    )


@app.get("/debug/personas", tags=["Debug"])
async def debug_personas():
    """Debug endpoint to check if persona files are accessible."""
    from pathlib import Path
    import os
    
    debug_info = {
        "working_directory": os.getcwd(),
        "siliconsampling_exists": Path("SiliconSampling").exists(),
        "personas_v2_exists": Path("SiliconSampling/personas_v2").exists(),
        "persona_files": {}
    }
    
    # Check each persona strategy
    for strategy in ["random", "wpp", "ipip"]:
        file_path = Path(f"SiliconSampling/personas_v2/{strategy}_matching/personas_{strategy}_v2.jsonl")
        debug_info["persona_files"][strategy] = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0
        }
        
        # Try to count lines
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                debug_info["persona_files"][strategy]["persona_count"] = line_count
            except Exception as e:
                debug_info["persona_files"][strategy]["error"] = str(e)
    
    return debug_info


@app.get("/debug/env", tags=["Debug"])
async def debug_environment():
    """Debug endpoint to check environment configuration."""
    import os
    
    return {
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "gemini_api_key_length": len(os.getenv("GEMINI_API_KEY", "")) if os.getenv("GEMINI_API_KEY") else 0,
        "pythonpath": os.getenv("PYTHONPATH", ""),
        "port": os.getenv("PORT", "8080")
    }


@app.get("/models", tags=["Info"])
async def list_models():
    """List model configuration and persona information."""
    model_info = {
        "unified_model": {
            "model": "gemini-2.5-flash-lite",
            "provider": "Google",
            "description": "Fastest and most cost-effective model with native multimodal support for click decisions and analysis",
            "capabilities": ["text", "images (native vision)", "analysis generation"],
            "env_var": "GEMINI_API_KEY"
        }
    }
    
    persona_info = {
        "v1": {
            "description": "Basic personas with simple description",
            "components": ["persona_description"]
        },
        "v2": {
            "description": "Enhanced personas with psychological depth",
            "components": ["persona_description", "behavioral_tendencies", "self_schema"]
        }
    }
    
    strategy_info = {
        "random": {
            "description": "Random OCEAN scores (baseline/control)",
            "ocean_source": "Random uniform distribution (0-10)"
        },
        "wpp": {
            "description": "WPP survey-based personality matching",
            "ocean_source": "Vietnamese WPP survey data (~1,055 respondents)"
        },
        "ipip": {
            "description": "IPIP demographic matching",
            "ocean_source": "IPIP-NEO Big Five dataset"
        }
    }
    
    return {
        "models": model_info,
        "persona_versions": persona_info,
        "persona_strategies": strategy_info,
        "platforms": AVAILABLE_PLATFORMS
    }


@app.post("/predict/text", response_model=CTRResponse, tags=["Prediction"])
async def predict_text_ad_ctr(request: CTRTextRequest):
    """
    Predict click-through rate for a text advertisement using synthetic personas.
    
    This endpoint:
    1. Loads synthetic personas based on specified version and strategy
    2. For each persona, gets LLM prediction of click decision + reasoning
    3. Aggregates results and generates final analysis with recommendations using Gemini
    
    **Parameters:**
    - **ad_text**: Advertisement text to evaluate
    - **population_size**: Number of personas to sample (1-10,000)
    - **ad_platform**: Platform context (facebook, tiktok, amazon, etc.)
    - **persona_version**: Use v1 (basic) or v2 (enhanced) personas
    - **persona_strategy**: OCEAN assignment strategy (random, wpp, ipip)
    - **concurrent_requests**: Parallelism level for API calls
    - **include_persona_details**: Include individual persona responses
    - **decision_model**: LLM model for decisions (default: gemini-2.5-flash)
    
    **Returns:**
    - Predicted CTR, click counts, runtime statistics
    - Final analysis with insights and recommendations
    - Optional: Individual persona responses with reasoning
    """
    try:
        # Validate inputs
        if request.ad_platform not in AVAILABLE_PLATFORMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ad_platform. Must be one of: {AVAILABLE_PLATFORMS}"
            )
        
        if request.persona_version not in AVAILABLE_PERSONA_VERSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid persona_version. Must be one of: {AVAILABLE_PERSONA_VERSIONS}"
            )
        
        if request.persona_strategy not in AVAILABLE_PERSONA_STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid persona_strategy. Must be one of: {AVAILABLE_PERSONA_STRATEGIES}"
            )
        
        # Create predictor
        try:
            predictor = CTRPredictor(
                persona_version=request.persona_version,
                persona_strategy=request.persona_strategy,
                concurrent_requests=request.concurrent_requests,
                realistic_mode=request.realistic_mode,
                decision_model=request.decision_model
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"Persona file not found. Please ensure personas are generated. Error: {str(e)}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration error: {str(e)}"
            )
        
        # Run prediction
        start_time = time.time()
        
        try:
            result: CTRPredictionResult = await predictor.predict_async(
                ad_text=request.ad_text,
                population_size=request.population_size,
                ad_platform=request.ad_platform
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {type(e).__name__}: {str(e)}"
            )
        
        runtime = time.time() - start_time
        
        # Build response
        response = CTRResponse(
            success=True,
            ctr=round(result.ctr, 4),
            total_clicks=result.total_clicks,
            total_personas=result.total_personas,
            runtime_seconds=round(runtime, 2),
            provider_used=result.provider_used,
            model_used=result.model_used,
            ad_platform=result.ad_platform,
            persona_version=request.persona_version,
            persona_strategy=request.persona_strategy,
            timestamp=datetime.utcnow().isoformat() + "Z",
            final_analysis=result.final_analysis
        )
        
        # Add persona details if requested
        if request.include_persona_details:
            response.persona_responses = [
                PersonaResponseModel(
                    persona_id=r.persona_id,
                    will_click=r.will_click,
                    reasoning=r.reasoning,
                    demographics=r.demographics,
                    **parse_demographics_string(r.demographics)
                )
                for r in result.persona_responses
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/image", response_model=CTRResponse, tags=["Prediction"])
async def predict_image_ad_ctr(request: CTRImageRequest):
    """
    Predict click-through rate for an image advertisement using synthetic personas.
    
    This endpoint:
    1. Loads synthetic personas based on specified version and strategy
    2. For each persona, gets LLM prediction of click decision + reasoning using Gemini (vision)
    3. Aggregates results and generates final analysis with recommendations using Gemini
    
    **Parameters:**
    - **image_url**: Advertisement image URL (http/https or data URL with base64)
    - **population_size**: Number of personas to sample (1-10,000)
    - **ad_platform**: Platform context (facebook, tiktok, amazon, etc.)
    - **persona_version**: Use v1 (basic) or v2 (enhanced) personas
    - **persona_strategy**: OCEAN assignment strategy (random, wpp, ipip)
    - **concurrent_requests**: Parallelism level for API calls
    - **include_persona_details**: Include individual persona responses
    
    **Returns:**
    - Predicted CTR, click counts, runtime statistics
    - Final analysis with insights and recommendations
    - Optional: Individual persona responses with reasoning
    """
    try:
        # Validate inputs
        if request.ad_platform not in AVAILABLE_PLATFORMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ad_platform. Must be one of: {AVAILABLE_PLATFORMS}"
            )
        
        if request.persona_version not in AVAILABLE_PERSONA_VERSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid persona_version. Must be one of: {AVAILABLE_PERSONA_VERSIONS}"
            )
        
        if request.persona_strategy not in AVAILABLE_PERSONA_STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid persona_strategy. Must be one of: {AVAILABLE_PERSONA_STRATEGIES}"
            )
        
        # Create predictor
        try:
            predictor = CTRPredictor(
                persona_version=request.persona_version,
                persona_strategy=request.persona_strategy,
                concurrent_requests=request.concurrent_requests,
                realistic_mode=request.realistic_mode
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"Persona file not found. Please ensure personas are generated. Error: {str(e)}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration error: {str(e)}"
            )
        
        # Run prediction
        start_time = time.time()
        
        try:
            result: CTRPredictionResult = await predictor.predict_async(
                image_url=request.image_url,
                population_size=request.population_size,
                ad_platform=request.ad_platform
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {type(e).__name__}: {str(e)}"
            )
        
        runtime = time.time() - start_time
        
        # Build response
        response = CTRResponse(
            success=True,
            ctr=round(result.ctr, 4),
            total_clicks=result.total_clicks,
            total_personas=result.total_personas,
            runtime_seconds=round(runtime, 2),
            provider_used=result.provider_used,
            model_used=result.model_used,
            ad_platform=result.ad_platform,
            persona_version=request.persona_version,
            persona_strategy=request.persona_strategy,
            timestamp=datetime.utcnow().isoformat() + "Z",
            final_analysis=result.final_analysis
        )
        
        # Add persona details if requested
        if request.include_persona_details:
            response.persona_responses = [
                PersonaResponseModel(
                    persona_id=r.persona_id,
                    will_click=r.will_click,
                    reasoning=r.reasoning,
                    demographics=r.demographics,
                    **parse_demographics_string(r.demographics)
                )
                for r in result.persona_responses
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Info"])


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error=f"Internal server error: {str(exc)}",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


if __name__ == "__main__":
    # Development server
    print("\n" + "="*80)
    print("WISTERIA CTR STUDIO API SERVER v3.1.0")
    print("="*80)
    print("\n📡 Starting server...")
    print("   Docs: http://localhost:8080/docs")
    print("   ReDoc: http://localhost:8080/redoc")
    print("   Health: http://localhost:8080/health")
    print("\n💡 Model:")
    print("   Unified: gemini-2.5-flash-lite (click decisions, analysis, text & images)")
    print("\n📚 Endpoints:")
    print("   POST /predict/text - Text ad prediction")
    print("   POST /predict/image - Image ad prediction")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
