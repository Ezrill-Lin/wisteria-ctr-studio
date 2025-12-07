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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from CTRPrediction import CTRPredictor, CTRPredictionResult


# Pydantic models for request/response
class CTRRequest(BaseModel):
    """Request model for CTR prediction."""
    ad_content: str = Field(
        description="Advertisement text content",
        min_length=1,
        example="Special 0% APR credit card offer for travel rewards"
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
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, deepseek, gemini)",
        pattern="^(openai|deepseek|gemini)$",
        example="openai"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name (uses provider default if not specified)",
        example="gpt-4o-mini"
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

    class Config:
        schema_extra = {
            "example": {
                "ad_content": "Special 0% APR credit card offer for travel rewards",
                "population_size": 100,
                "ad_platform": "facebook",
                "persona_version": "v2",
                "persona_strategy": "random",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "concurrent_requests": 10,
                "include_persona_details": False
            }
        }


class PersonaResponseModel(BaseModel):
    """Individual persona response."""
    persona_id: str
    will_click: bool
    reasoning: str
    demographics: str = ""


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
                "provider_used": "openai",
                "model_used": "gpt-4o-mini",
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
    available_providers: List[str]
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
    description="REST API for CTR prediction using SiliconSampling synthetic personas and LLM-based behavioral simulation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Configuration
AVAILABLE_PROVIDERS = ["openai", "deepseek", "gemini"]
AVAILABLE_PERSONA_VERSIONS = ["v1", "v2"]
AVAILABLE_PERSONA_STRATEGIES = ["random", "wpp", "ipip"]
AVAILABLE_PLATFORMS = ["facebook", "tiktok", "amazon", "instagram", "youtube"]


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Wisteria CTR Studio API",
        "version": "2.0.0",
        "description": "REST API for CTR prediction using SiliconSampling personas",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "providers": "/providers",
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
        version="2.0.0",
        available_providers=AVAILABLE_PROVIDERS,
        available_persona_versions=AVAILABLE_PERSONA_VERSIONS,
        available_persona_strategies=AVAILABLE_PERSONA_STRATEGIES,
        available_platforms=AVAILABLE_PLATFORMS
    )


@app.get("/providers", tags=["Info"])
async def list_providers():
    """List available LLM providers and persona configurations."""
    provider_info = {
        "openai": {
            "default_model": "gpt-4o-mini",
            "description": "OpenAI GPT models (recommended for quality)",
            "env_var": "OPENAI_API_KEY"
        },
        "deepseek": {
            "default_model": "deepseek-chat",
            "description": "DeepSeek models (cost-effective)",
            "env_var": "DEEPSEEK_API_KEY"
        },
        "gemini": {
            "default_model": "gemini-2.0-flash-exp",
            "description": "Google Gemini models (fast and efficient)",
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
        "providers": provider_info,
        "persona_versions": persona_info,
        "persona_strategies": strategy_info,
        "platforms": AVAILABLE_PLATFORMS
    }


@app.post("/predict", response_model=CTRResponse, tags=["Prediction"])
async def predict_ctr(request: CTRRequest):
    """
    Predict click-through rate for an advertisement using synthetic personas.
    
    This endpoint:
    1. Loads synthetic personas based on specified version and strategy
    2. For each persona, gets LLM prediction of click decision + reasoning
    3. Aggregates results and generates final analysis with recommendations
    
    **Parameters:**
    - **ad_content**: Advertisement text to evaluate
    - **population_size**: Number of personas to sample (1-10,000)
    - **ad_platform**: Platform context (facebook, tiktok, amazon, etc.)
    - **persona_version**: Use v1 (basic) or v2 (enhanced) personas
    - **persona_strategy**: OCEAN assignment strategy (random, wpp, ipip)
    - **provider**: LLM provider to use
    - **model**: Specific model name (optional, uses provider default)
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
        
        if request.provider not in AVAILABLE_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Must be one of: {AVAILABLE_PROVIDERS}"
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
                provider=request.provider,
                model=request.model,
                concurrent_requests=request.concurrent_requests
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
                ad_content=request.ad_content,
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
                    demographics=r.demographics
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
    print("WISTERIA CTR STUDIO API SERVER")
    print("="*80)
    print("\n📡 Starting server...")
    print("   Docs: http://localhost:8080/docs")
    print("   ReDoc: http://localhost:8080/redoc")
    print("   Health: http://localhost:8080/health")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
