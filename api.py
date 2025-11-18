"""FastAPI web service for CTR simulation.

This module provides a REST API interface for the Wisteria CTR Studio,
allowing users to predict click-through rates via HTTP requests.
"""

import os
import time
import json
import io
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Google Cloud Storage for identity bank loading
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from SiliconSampling.sampler import load_identity_bank, sample_identities
from CTRPrediction.llm_click_model import LLMClickPredictor, DistributedLLMPredictor


# Pydantic models for request/response validation
class CTRRequest(BaseModel):
    """Request model for CTR prediction."""
    ad_text: str = Field(default="Special 0% APR credit card offer for travel rewards", description="Advertisement text to evaluate", min_length=1)
    ad_platform: str = Field(default="facebook", description="Platform where ad is shown")
    population_size: int = Field(default=1000, description="Number of identities to sample", ge=1, le=1e8)
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    use_distributed: bool = Field(default=True, description="Use distributed load balancing (recommended)")
    provider: str = Field(default="openai", description="LLM provider to use (ignored if use_distributed=True)")
    model: Optional[str] = Field(default="gpt-4o-mini", description="Model name (uses provider default if not specified)")
    batch_size: int = Field(default=50, description="Batch size per LLM call", ge=1, le=200)
    use_mock: bool = Field(default=False, description="Force mock predictions")
    use_sync: bool = Field(default=False, description="Use synchronous processing")
    api_key: Optional[str] = Field(default=None, description="API key override")
    identity_bank_path: Optional[str] = Field(default="", description="Custom identity bank path")

    class Config:
        schema_extra = {
            "example": {
                "ad_text": "Special 0% APR credit card offer for travel rewards",
                "ad_platform": "facebook",
                "population_size": 1000,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "use_mock": False
            }
        }


class IdentityProfile(BaseModel):
    """Individual identity profile model."""
    gender: str
    age: int
    region: str
    occupation: str
    annual_salary: float
    liability_status: float
    is_married: bool
    health_status: bool
    illness: Optional[str] = None


class DetailedResult(BaseModel):
    """Detailed prediction result for a single identity."""
    id: int
    profile: IdentityProfile
    click_prediction: int


class CTRResponse(BaseModel):
    """Response model for CTR prediction."""
    success: bool
    ctr: float
    total_clicks: int
    total_identities: int
    runtime_seconds: float
    provider_used: str
    model_used: str
    processing_mode: str
    ad_platform: str
    timestamp: str
    detailed_results: Optional[List[DetailedResult]] = None

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "ctr": 0.247,
                "total_clicks": 247,
                "total_identities": 1000,
                "runtime_seconds": 12.34,
                "provider_used": "openai",
                "model_used": "gpt-4o-mini",
                "processing_mode": "asynchronous parallel",
                "ad_platform": "facebook",
                "timestamp": "2025-10-26T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    available_providers: List[str]
    version: str


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool
    error: str
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Wisteria CTR Studio API",
    description="REST API for click-through rate prediction using synthetic identities and LLM models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global configuration
DEFAULT_IDENTITY_BANK_PATH = os.path.join("SiliconSampling", "data", "identity_bank.json")
AVAILABLE_PROVIDERS = ["openai", "deepseek", "runpod"]
AVAILABLE_PLATFORMS = ["facebook", "tiktok", "amazon"]

# Google Cloud Storage configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "wisteria-data-bucket")
GCS_IDENTITY_BANK_PATH = os.getenv("GCS_IDENTITY_BANK_PATH", "data/identity_bank.json")

# Application state for caching identity bank
app.state.identity_bank = None
app.state.gcs_client = None


def load_identity_bank_from_gcs():
    """Load identity bank from Google Cloud Storage."""
    if not GCS_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Google Cloud Storage client not available. Install with: pip install google-cloud-storage"
        )
    
    try:
        if app.state.gcs_client is None:
            app.state.gcs_client = storage.Client()
        
        bucket = app.state.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_IDENTITY_BANK_PATH)
        
        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Identity bank file not found in GCS: gs://{GCS_BUCKET_NAME}/{GCS_IDENTITY_BANK_PATH}"
            )
        
        content = blob.download_as_text()
        return json.loads(content)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load identity bank from GCS: {str(e)}"
        )


def get_identity_bank(identity_bank_path: Optional[str] = None):
    """Get identity bank from cache, local file, or GCS."""
    # If no specific path requested, try cached version first (most common case)
    if identity_bank_path is None:
        if app.state.identity_bank is not None:
            return app.state.identity_bank
        # Fall through to default loading logic below
    
    # If a specific path is requested, handle GCS URLs and local files
    if identity_bank_path and identity_bank_path != DEFAULT_IDENTITY_BANK_PATH:
        # Handle GCS URLs
        if identity_bank_path.startswith("gs://"):
            if not GCS_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="Google Cloud Storage client not available. Install with: pip install google-cloud-storage"
                )
            try:
                # Parse GCS URL: gs://bucket/path
                parts = identity_bank_path[5:].split('/', 1)  # Remove 'gs://' and split
                if len(parts) != 2:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid GCS URL format: {identity_bank_path}"
                    )
                
                bucket_name, blob_path = parts
                
                if app.state.gcs_client is None:
                    app.state.gcs_client = storage.Client()
                
                bucket = app.state.gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                if not blob.exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Identity bank file not found in GCS: {identity_bank_path}"
                    )
                
                content = blob.download_as_text()
                return json.loads(content)
                
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load identity bank from GCS: {str(e)}"
                )
        
        # Handle local file paths
        if not os.path.exists(identity_bank_path):
            raise HTTPException(
                status_code=400,
                detail=f"Identity bank file not found: {identity_bank_path}"
            )
        return load_identity_bank(identity_bank_path)
    
    # Try to use cached version first
    if app.state.identity_bank is not None:
        return app.state.identity_bank
    
    # Try loading from local file
    if os.path.exists(DEFAULT_IDENTITY_BANK_PATH):
        try:
            return load_identity_bank(DEFAULT_IDENTITY_BANK_PATH)
        except Exception:
            pass  # Fall back to GCS
    
    # Load from GCS if local file not available
    if GCS_AVAILABLE:
        try:
            return load_identity_bank_from_gcs()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load identity bank from any source: {str(e)}"
            )
    
    # No identity bank available
    raise HTTPException(
        status_code=500,
        detail="No identity bank available. Please ensure either local file exists or GCS is configured."
    )


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        # Try to preload identity bank for better performance
        if GCS_AVAILABLE:
            try:
                app.state.identity_bank = load_identity_bank_from_gcs()
                print(f"✅ Identity bank loaded from GCS: gs://{GCS_BUCKET_NAME}/{GCS_IDENTITY_BANK_PATH}")
            except Exception as e:
                print(f"⚠️  Could not preload from GCS: {e}")
        
        # Fallback to local file
        if app.state.identity_bank is None and os.path.exists(DEFAULT_IDENTITY_BANK_PATH):
            try:
                app.state.identity_bank = load_identity_bank(DEFAULT_IDENTITY_BANK_PATH)
                print(f"✅ Identity bank loaded from local file: {DEFAULT_IDENTITY_BANK_PATH}")
            except Exception as e:
                print(f"⚠️  Could not load local identity bank: {e}")
        
        if app.state.identity_bank is None:
            print("⚠️  No identity bank loaded at startup. Will attempt to load on first request.")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    app.state.identity_bank = None
    app.state.gcs_client = None


def compute_ctr(clicks: List[int]) -> float:
    """Compute click-through rate from binary predictions."""
    if not clicks:
        return 0.0
    return sum(1 for x in clicks if x) / float(len(clicks))


def validate_request(request: CTRRequest) -> None:
    """Validate request parameters."""
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


def get_default_model(provider: str) -> str:
    """Get default model for a provider."""
    defaults = {
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "runpod": "llama3.1-8b",
    }
    return defaults.get(provider, "gpt-4o-mini")


def format_identity_profile(profile: Dict[str, Any]) -> IdentityProfile:
    """Convert identity dictionary to IdentityProfile model."""
    return IdentityProfile(
        gender=profile.get("gender", ""),
        age=profile.get("age", 0),
        region=profile.get("region", ""),
        occupation=profile.get("occupation", ""),
        annual_salary=profile.get("annual_salary", 0.0),
        liability_status=profile.get("liability_status", 0.0),
        is_married=profile.get("is_married", False),
        health_status=profile.get("health_status", False),
        illness=profile.get("illness")
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        available_providers=AVAILABLE_PROVIDERS,
        version="1.0.0"
    )


@app.post("/predict-ctr", response_model=CTRResponse)
async def predict_ctr(request: CTRRequest, include_details: bool = False):
    """
    Predict click-through rate for an advertisement.
    
    Args:
        request: CTR prediction request parameters
        include_details: Whether to include detailed per-identity results
        
    Returns:
        CTR prediction results with summary statistics
    """
    try:
        # Validate request
        validate_request(request)
        
        # Load identity bank (from cache, local file, or GCS)
        bank = get_identity_bank(request.identity_bank_path)
        
        # Generate synthetic identities
        identities = sample_identities(
            request.population_size, 
            bank, 
            seed=request.seed
        )
        
        # Determine which predictor to use
        if request.use_distributed:
            # Use distributed load balancing
            predictor = DistributedLLMPredictor(
                batch_size=request.batch_size,
                use_mock=request.use_mock,
                providers=["openai", "deepseek", "gemini"],
            )
            provider_used = "distributed"
            model_used = "openai+deepseek+gemini"
        else:
            # Get model name for single provider
            model = request.model or get_default_model(request.provider)
            
            # Initialize single-provider predictor
            predictor = LLMClickPredictor(
                provider=request.provider,
                model=model,
                batch_size=request.batch_size,
                use_mock=request.use_mock,
                api_key=request.api_key,
            )
            provider_used = request.provider
            model_used = model
        
        # Predict clicks (always use async)
        start_time = time.time()
        clicks = await predictor.predict_clicks_async(
            request.ad_text, 
            identities, 
            request.ad_platform
        )
        end_time = time.time()
        runtime = end_time - start_time
        
        # Compute CTR
        ctr = compute_ctr(clicks)
        
        # Prepare response
        processing_mode = "distributed async" if request.use_distributed else "async parallel"
        
        response = CTRResponse(
            success=True,
            ctr=round(ctr, 4),
            total_clicks=sum(clicks),
            total_identities=len(identities),
            runtime_seconds=round(runtime, 2),
            provider_used=provider_used,
            model_used=model_used,
            processing_mode=processing_mode,
            ad_platform=request.ad_platform,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Add detailed results if requested
        if include_details:
            detailed_results = []
            for i, (profile, click) in enumerate(zip(identities, clicks)):
                detailed_results.append(DetailedResult(
                    id=i,
                    profile=format_identity_profile(profile),
                    click_prediction=click
                ))
            response.detailed_results = detailed_results
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict-ctr-batch")
async def predict_ctr_batch(requests: List[CTRRequest]):
    """
    Predict CTR for multiple advertisements in batch.
    
    Args:
        requests: List of CTR prediction requests
        
    Returns:
        List of CTR prediction results
    """
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 10 requests"
        )
    
    results = []
    for req in requests:
        try:
            result = await predict_ctr(req, include_details=False)
            results.append(result)
        except Exception as e:
            results.append(ErrorResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow().isoformat() + "Z"
            ))
    
    return results


@app.get("/providers")
async def list_providers():
    """List available LLM providers and their default models."""
    provider_info = {
        "openai": {
            "default_model": "gpt-4o-mini",
            "description": "OpenAI GPT models",
            "env_var": "OPENAI_API_KEY"
        },
        "deepseek": {
            "default_model": "deepseek-chat", 
            "description": "DeepSeek models",
            "env_var": "DEEPSEEK_API_KEY"
        },
        "runpod": {
            "default_model": "llama3.1-8b",
            "description": "RunPod-hosted vLLM (OpenAI-compatible)",
            "env_var": "RUNPOD_API_KEY (for serverless jobs), RUNPOD_LLAMA_8B_ENDPOINT / RUNPOD_LLAMA_70B_ENDPOINT (jobs) or RUNPOD_LLAMA_8B_URL / RUNPOD_LLAMA_70B_URL (HTTP)"
        }
    }
    
    return {
        "available_providers": provider_info,
        "platforms": AVAILABLE_PLATFORMS
    }


@app.get("/identities")
async def get_identities():
    """Get the identity bank configuration.
    
    Returns the complete identity bank structure including all categories,
    distributions, and sampling parameters.
    """
    try:
        bank = get_identity_bank()
        return {
            "success": True,
            "identity_bank": bank,
            "source": "gcs" if app.state.identity_bank and GCS_AVAILABLE else "local",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve identity bank: {str(e)}"
        )


@app.post("/identities/reload")
async def reload_identities():
    """Reload the identity bank from the data source.
    
    Forces a fresh load of the identity bank, bypassing any cached version.
    Useful for updating the configuration without restarting the service.
    """
    try:
        # Clear cached version
        app.state.identity_bank = None
        
        # Try to reload from GCS first
        if GCS_AVAILABLE:
            try:
                app.state.identity_bank = load_identity_bank_from_gcs()
                return {
                    "success": True,
                    "message": "Identity bank reloaded from Google Cloud Storage",
                    "source": "gcs",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            except Exception as e:
                print(f"Failed to reload from GCS: {e}")
        
        # Fallback to local file
        if os.path.exists(DEFAULT_IDENTITY_BANK_PATH):
            app.state.identity_bank = load_identity_bank(DEFAULT_IDENTITY_BANK_PATH)
            return {
                "success": True,
                "message": "Identity bank reloaded from local file",
                "source": "local",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        raise HTTPException(
            status_code=404,
            detail="No identity bank source available for reload"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload identity bank: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Wisteria CTR Studio API",
        "version": "1.0.0",
        "description": "REST API for click-through rate prediction using synthetic identities and LLM models",
        "endpoints": {
            "health": "/health",
            "predict": "/predict-ctr",
            "batch_predict": "/predict-ctr-batch", 
            "providers": "/providers",
            "identities": "/identities",
            "reload_identities": "/identities/reload",
            "docs": "/docs"
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
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
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )