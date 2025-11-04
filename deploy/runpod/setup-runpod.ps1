# RunPod Configuration for Wisteria CTR Studio (PowerShell)
# This script sets up RunPod serverless endpoints for cost-effective auto-scaling

Write-Host ">>> Setting up RunPod serverless endpoints for Wisteria CTR Studio" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# Check for API key
if (-not $env:RUNPOD_API_KEY) {
    Write-Host "[ERROR] RUNPOD_API_KEY environment variable is not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "Setup steps:" -ForegroundColor Yellow
    Write-Host "1. Go to https://runpod.io/console/user/settings"
    Write-Host "2. Generate an API key"
    Write-Host "3. Run: `$env:RUNPOD_API_KEY='your-api-key-here'"
    Write-Host "4. Re-run this script"
    Write-Host ""
    exit 1
}

$apiKeyPreview = $env:RUNPOD_API_KEY.Substring(0, [Math]::Min(8, $env:RUNPOD_API_KEY.Length))
Write-Host "[OK] RunPod API key found: $apiKeyPreview..." -ForegroundColor Green

# Function to create serverless endpoint
function Create-Endpoint {
    param(
        [string]$Name,
        [string]$Model,
        [string]$Gpu,
        [int]$MaxWorkers
    )
    
    Write-Host "[INFO] Creating $Name endpoint..." -ForegroundColor Blue
    
    $body = @{
        name = $Name
        template = @{
            imageName = "vllm/vllm-openai:latest"
            containerConfig = @{
                arguments = @(
                    "--model=$Model",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--trust-remote-code",
                    "--download-dir=/workspace/models"
                )
                ports = @("8000/http")
            }
            volumeInGb = 80
            containerDiskInGb = 30
            env = @(
                @{key = "HF_HOME"; value = "/workspace/cache"}
                @{key = "TRANSFORMERS_CACHE"; value = "/workspace/models"}
            )
        }
        workersMin = 0
        workersMax = $MaxWorkers
        gpuTypeId = $Gpu
        scalerType = "QUEUE_DELAY"
        scalerSettings = @{
            queueDelay = 5
            maxIdleTime = 300
        }
    } | ConvertTo-Json -Depth 10
    
    try {
        # RunPod uses GraphQL API for management operations
        $query = @{
            query = @"
                mutation {
                    saveTemplate(input: {
                        name: "$Name-template"
                        imageName: "vllm/vllm-openai:latest"
                        startJupyter: false
                        startSsh: false
                        volumeInGb: 80
                        containerDiskInGb: 30
                        env: [
                            {key: "HF_HOME", value: "/workspace/cache"}
                            {key: "TRANSFORMERS_CACHE", value: "/workspace/models"}
                        ]
                        dockerArgs: "--model=$Model --host=0.0.0.0 --port=8000 --trust-remote-code --download-dir=/workspace/models"
                        ports: "8000/http"
                    }) {
                        id
                        name
                    }
                }
"@
        } | ConvertTo-Json
        
        Write-Host "Creating template first..." -ForegroundColor Yellow
        $templateResponse = Invoke-RestMethod -Uri "https://api.runpod.ai/graphql" `
            -Method POST `
            -Headers @{
                "Authorization" = "Bearer $env:RUNPOD_API_KEY"
                "Content-Type" = "application/json"
            } `
            -Body $query
        
        if ($templateResponse.errors) {
            Write-Host "[ERROR] Template creation failed: $($templateResponse.errors)" -ForegroundColor Red
            return $null
        }
        
        Write-Host "[OK] Template created, now creating endpoint..." -ForegroundColor Green
        
        # Note: Endpoint creation via API might require different approach
        # For now, provide manual instructions
        Write-Host "[WARNING] Please create the endpoint manually in RunPod console:" -ForegroundColor Yellow
        Write-Host "   1. Go to: https://runpod.io/console/serverless" -ForegroundColor Cyan
        Write-Host "   2. Click 'New Endpoint'" -ForegroundColor Cyan
        Write-Host "   3. Select template: $Name-template" -ForegroundColor Cyan
        Write-Host "   4. Configure GPU: $Gpu" -ForegroundColor Cyan
        Write-Host "   5. Set Max Workers: $MaxWorkers" -ForegroundColor Cyan
        Write-Host ""
        
        return "manual-setup-required"
    }
    catch {
        Write-Host "[ERROR] Error creating $Name endpoint: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Since RunPod API endpoints have changed, let's provide guided manual setup
Write-Host ""
Write-Host "*** Manual Setup Required ***" -ForegroundColor Yellow
Write-Host "The RunPod API has changed. Please follow these steps:" -ForegroundColor Yellow
Write-Host ""

# Provide step-by-step manual instructions
Write-Host "STEP 1: Create Llama 8B Endpoint" -ForegroundColor Cyan
Write-Host "1. Go to: https://runpod.io/console/serverless"
Write-Host "2. Click 'New Endpoint'"
Write-Host "3. Use these settings:"
Write-Host "   - Name: wisteria-llama-8b"
Write-Host "   - Template: vllm/vllm-openai:latest"
Write-Host "   - Model: meta-llama/Llama-3.1-8B-Instruct"
Write-Host "   - GPU: RTX 4090"
Write-Host "   - Container Start Command:"
Write-Host "     python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-3.1-8B-Instruct --host=0.0.0.0 --port=8000 --trust-remote-code"
Write-Host "   - Container Disk: 30 GB"
Write-Host "   - Volume: 80 GB"
Write-Host "   - Max Workers: 3"
Write-Host ""

Write-Host "STEP 2: Create Llama 70B Endpoint" -ForegroundColor Cyan  
Write-Host "1. Create another endpoint with:"
Write-Host "   - Name: wisteria-llama-70b"
Write-Host "   - Model: meta-llama/Llama-3.1-70B-Instruct"
Write-Host "   - GPU: A100 40GB"
Write-Host "   - Container Start Command:"
Write-Host "     python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-3.1-70B-Instruct --host=0.0.0.0 --port=8000 --trust-remote-code --tensor-parallel-size=2"
Write-Host "   - Container Disk: 40 GB"
Write-Host "   - Volume: 150 GB"
Write-Host "   - Max Workers: 1"
Write-Host ""

Write-Host "STEP 3: Set Environment Variables" -ForegroundColor Cyan
Write-Host "After creating endpoints, copy their IDs and run:"
Write-Host "`$env:RUNPOD_LLAMA_8B_ENDPOINT='your-8b-endpoint-id'"
Write-Host "`$env:RUNPOD_LLAMA_70B_ENDPOINT='your-70b-endpoint-id'"
Write-Host ""

Write-Host "STEP 4: Test Setup" -ForegroundColor Cyan
Write-Host "cd ..\.."
Write-Host "python demo.py --provider runpod --runpod-model llama-8b --ad 'Test ad' --population-size 10"
Write-Host ""

# Create a simplified config template
$configTemplate = @"
# RunPod Configuration Template
# Replace 'your-endpoint-id' with actual endpoint IDs from RunPod console

`$env:RUNPOD_API_KEY="$env:RUNPOD_API_KEY"
`$env:RUNPOD_LLAMA_8B_ENDPOINT="your-8b-endpoint-id-here"
`$env:RUNPOD_LLAMA_70B_ENDPOINT="your-70b-endpoint-id-here"

Write-Host ">>> RunPod environment configured!" -ForegroundColor Green
Write-Host "Test with: python demo.py --provider runpod --runpod-model llama-8b --ad 'Test ad' --population-size 100" -ForegroundColor Yellow
"@

$configTemplate | Out-File -FilePath "runpod-config-template.ps1" -Encoding UTF8

Write-Host "Why No HuggingFace API Key Needed:" -ForegroundColor Green
Write-Host "[OK] Llama 3.1 models are publicly available"
Write-Host "[OK] No gated access restrictions"  
Write-Host "[OK] Models download automatically on first use"
Write-Host "[OK] Cached in persistent volume for faster subsequent starts"
Write-Host ""

Write-Host "Tips:" -ForegroundColor Yellow
Write-Host "- First model load takes 5-10 minutes (downloads ~16GB for 8B, ~140GB for 70B)"
Write-Host "- Subsequent starts are much faster (~30-90 seconds)"
Write-Host "- Use larger volumes to cache multiple models"
Write-Host "- Monitor costs at: https://runpod.io/console/billing"