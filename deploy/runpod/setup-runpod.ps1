# RunPod vLLM Serverless Deployment Script for Wisteria CTR Studio
# 
# This script automatically creates RunPod serverless endpoints using GraphQL API
# 
# Usage:
#   1. Set your RunPod API key: $env:RUNPOD_API_KEY="your-key-here"
#   2. Run: .\setup-runpod.ps1
#   3. Source the generated config: .\runpod-config.ps1
#   4. Test: cd ..\.. ; python demo.py --provider runpod --runpod-model llama-8b

Write-Host "===================================================" -ForegroundColor Green
Write-Host "  RunPod vLLM Serverless Deployment" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""

# Check for API key
if (-not $env:RUNPOD_API_KEY) {
    Write-Host "[ERROR] RUNPOD_API_KEY environment variable is not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please get your API key from https://runpod.io/console/user/settings and run:" -ForegroundColor Yellow
    Write-Host '  $env:RUNPOD_API_KEY="your-key-here"' -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

$apiKeyPreview = $env:RUNPOD_API_KEY.Substring(0, [Math]::Min(8, $env:RUNPOD_API_KEY.Length))
Write-Host "[OK] RunPod API key found: $apiKeyPreview..." -ForegroundColor Green
Write-Host ""


# GraphQL API endpoint
$graphqlUrl = "https://api.runpod.io/graphql?api_key=$env:RUNPOD_API_KEY"
$headers = @{
    "Content-Type" = "application/json"
}

# Helper function to execute GraphQL mutations
function Invoke-RunPodGraphQL {
    param(
        [string]$Query
    )
    
    $body = @{
        query = $Query
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri $graphqlUrl -Method POST -Headers $headers -Body $body
        return $response
    } catch {
        Write-Host "[ERROR] GraphQL Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Response: $($_.Exception.Response)" -ForegroundColor Red
        return $null
    }
}

# Template configurations
$templates = @{
    "llama-8b" = @{
        name = "wisteria-vllm-llama-8b"
        imageName = "vllm/vllm-openai:latest"
        dockerArgs = "python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-3.1-8B-Instruct --host=0.0.0.0 --port=8000 --trust-remote-code --download-dir=/runpod-volume/cache --max-model-len=4096 --gpu-memory-utilization=0.9"
        volumeInGb = 80
        containerDiskInGb = 30
        ports = "8000/http"
        volumeMountPath = "/runpod-volume"
    }
    "llama-70b" = @{
        name = "wisteria-vllm-llama-70b"
        imageName = "vllm/vllm-openai:latest"
        dockerArgs = "python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-3.1-70B-Instruct --host=0.0.0.0 --port=8000 --trust-remote-code --download-dir=/runpod-volume/cache --max-model-len=4096 --gpu-memory-utilization=0.95 --tensor-parallel-size=2"
        volumeInGb = 150
        containerDiskInGb = 50
        ports = "8000/http"
        volumeMountPath = "/runpod-volume"
    }
}

# Endpoint configurations
$endpoints = @{
    "llama-8b" = @{
        templateId = ""  # Will be filled after template creation
        gpuTypeId = "NVIDIA RTX 4090"
        workersMin = 0
        workersMax = 3
        idleTimeout = 5
    }
    "llama-70b" = @{
        templateId = ""
        gpuTypeId = "NVIDIA A100-PCIE-40GB"
        workersMin = 0
        workersMax = 1
        idleTimeout = 5
    }
}

$createdEndpoints = @{}

# Step 1: Create Templates
Write-Host "[Step 1] Creating vLLM Templates..." -ForegroundColor Cyan
Write-Host ""

foreach ($key in $templates.Keys) {
    $template = $templates[$key]
    Write-Host "  Creating template: $($template.name)..." -ForegroundColor Yellow
    
    $mutation = @"
mutation {
  saveTemplate(input: {
    name: "$($template.name)"
    imageName: "$($template.imageName)"
    dockerArgs: "$($template.dockerArgs)"
    volumeInGb: $($template.volumeInGb)
    containerDiskInGb: $($template.containerDiskInGb)
    ports: "$($template.ports)"
    volumeMountPath: "$($template.volumeMountPath)"
    startJupyter: false
    startSsh: false
    env: [
      {key: "HF_HOME", value: "/runpod-volume/cache"}
      {key: "TRANSFORMERS_CACHE", value: "/runpod-volume/cache"}
    ]
  }) {
    id
    name
  }
}
"@
    
    $result = Invoke-RunPodGraphQL -Query $mutation
    
    if ($result -and $result.data -and $result.data.saveTemplate) {
        $templateId = $result.data.saveTemplate.id
        $endpoints[$key].templateId = $templateId
        Write-Host "  [OK] Template created with ID: $templateId" -ForegroundColor Green
    } elseif ($result -and $result.errors) {
        Write-Host "  [ERROR] Failed to create template: $($result.errors[0].message)" -ForegroundColor Red
        
        # Try to find existing template
        Write-Host "  [INFO] Searching for existing template..." -ForegroundColor Yellow
        $queryTemplates = @"
query {
  myself {
    podTemplates {
      id
      name
    }
  }
}
"@
        $existingResult = Invoke-RunPodGraphQL -Query $queryTemplates
        if ($existingResult -and $existingResult.data -and $existingResult.data.myself) {
            $existingTemplate = $existingResult.data.myself.podTemplates | Where-Object { $_.name -eq $template.name }
            if ($existingTemplate) {
                $endpoints[$key].templateId = $existingTemplate.id
                Write-Host "  [OK] Found existing template with ID: $($existingTemplate.id)" -ForegroundColor Green
            } else {
                Write-Host "  [WARNING] No existing template found. You may need to create it manually." -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  [ERROR] Unexpected response from API" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 2
}

Write-Host ""

# Step 2: Create Serverless Endpoints
Write-Host "[Step 2] Creating Serverless Endpoints..." -ForegroundColor Cyan
Write-Host ""

foreach ($key in $endpoints.Keys) {
    $endpoint = $endpoints[$key]
    $template = $templates[$key]
    
    if (-not $endpoint.templateId) {
        Write-Host "  [WARNING] Skipping $($template.name): No template ID available" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "  Creating endpoint: $($template.name)..." -ForegroundColor Yellow
    
    $mutation = @"
mutation {
  saveEndpoint(input: {
    name: "$($template.name)"
    templateId: "$($endpoint.templateId)"
    gpuIds: "$($endpoint.gpuTypeId)"
    networkVolumeId: ""
    locations: ""
    idleTimeout: $($endpoint.idleTimeout)
    scalerType: "QUEUE_DELAY"
    scalerValue: 4
    workersMin: $($endpoint.workersMin)
    workersMax: $($endpoint.workersMax)
  }) {
    id
    name
    gpuIds
    workersMin
    workersMax
  }
}
"@
    
    $result = Invoke-RunPodGraphQL -Query $mutation
    
    if ($result -and $result.data -and $result.data.saveEndpoint) {
        $endpointId = $result.data.saveEndpoint.id
        $createdEndpoints[$key] = $endpointId
        Write-Host "  [OK] Endpoint created with ID: $endpointId" -ForegroundColor Green
    } elseif ($result -and $result.errors) {
        Write-Host "  [ERROR] Failed to create endpoint: $($result.errors[0].message)" -ForegroundColor Red
        
        # Try to find existing endpoint
        Write-Host "  [INFO] Searching for existing endpoint..." -ForegroundColor Yellow
        $queryEndpoints = @"
query {
  myself {
    endpoints {
      id
      name
    }
  }
}
"@
        $existingResult = Invoke-RunPodGraphQL -Query $queryEndpoints
        if ($existingResult -and $existingResult.data -and $existingResult.data.myself) {
            $existingEndpoint = $existingResult.data.myself.endpoints | Where-Object { $_.name -eq $template.name }
            if ($existingEndpoint) {
                $createdEndpoints[$key] = $existingEndpoint.id
                Write-Host "  [OK] Found existing endpoint with ID: $($existingEndpoint.id)" -ForegroundColor Green
            } else {
                Write-Host "  [WARNING] No existing endpoint found." -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  [ERROR] Unexpected response from API" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 2
}

Write-Host ""

# Step 3: Generate Configuration File
Write-Host "[Step 3] Generating Configuration..." -ForegroundColor Cyan
Write-Host ""

if ($createdEndpoints.Count -eq 0) {
    Write-Host "[ERROR] No endpoints were created. Please check the errors above." -ForegroundColor Red
    Write-Host ""
    Write-Host "You can manually create endpoints at: https://runpod.io/console/serverless" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$configContent = @"
# Wisteria CTR Studio - RunPod Configuration
# Generated on $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
#
# Source this file to configure your environment:
#   .\runpod-config.ps1

"@

foreach ($key in $createdEndpoints.Keys) {
    $endpointId = $createdEndpoints[$key]
    # RunPod serverless endpoints use the format: https://api.runpod.ai/v2/{endpoint_id}
    # For OpenAI-compatible vLLM, we need the direct URL format
    $url = "https://api.runpod.ai/v2/$endpointId/openai/v1"
    
    if ($key -eq "llama-8b") {
        $configContent += "`$env:RUNPOD_LLAMA_8B_URL=`"$url`"`n"
        Write-Host "  [OK] Llama 8B endpoint: $endpointId" -ForegroundColor Green
    } elseif ($key -eq "llama-70b") {
        $configContent += "`$env:RUNPOD_LLAMA_70B_URL=`"$url`"`n"
        Write-Host "  [OK] Llama 70B endpoint: $endpointId" -ForegroundColor Green
    }
}

$configContent += @"

Write-Host "[OK] RunPod environment configured!" -ForegroundColor Green
Write-Host ""
Write-Host "Endpoint URLs:" -ForegroundColor Cyan
if (`$env:RUNPOD_LLAMA_8B_URL) { Write-Host "  8B:  `$env:RUNPOD_LLAMA_8B_URL" }
if (`$env:RUNPOD_LLAMA_70B_URL) { Write-Host "  70B: `$env:RUNPOD_LLAMA_70B_URL" }
Write-Host ""
Write-Host "Test with:" -ForegroundColor Yellow
Write-Host "  python demo.py --provider runpod --runpod-model llama-8b --population-size 100" -ForegroundColor Cyan
"@

Set-Content -Path "runpod-config.ps1" -Value $configContent -Encoding UTF8

Write-Host ""
Write-Host "[SUCCESS] Deployment Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Wait 5-10 minutes for initial cold start (model download)" -ForegroundColor White
Write-Host "  2. Load the configuration:" -ForegroundColor White
Write-Host "       .\runpod-config.ps1" -ForegroundColor Cyan
Write-Host "  3. Test the deployment:" -ForegroundColor White
Write-Host "       cd ..\.. ; python demo.py --provider runpod --runpod-model llama-8b --population-size 100" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor your endpoints at: https://runpod.io/console/serverless" -ForegroundColor Yellow
Write-Host ""
