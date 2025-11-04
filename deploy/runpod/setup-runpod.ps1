# RunPod Configuration for Wisteria CTR Studio (PowerShell)
# This script sets up RunPod serverless endpoints for cost-effective auto-scaling

Write-Host "🚀 Setting up RunPod serverless endpoints for Wisteria CTR Studio" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# Check for API key
if (-not $env:RUNPOD_API_KEY) {
    Write-Host "❌ RUNPOD_API_KEY environment variable is not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "📋 Setup steps:" -ForegroundColor Yellow
    Write-Host "1. Go to https://runpod.io/console/user/settings"
    Write-Host "2. Generate an API key"
    Write-Host "3. Run: `$env:RUNPOD_API_KEY='your-api-key-here'"
    Write-Host "4. Re-run this script"
    Write-Host ""
    exit 1
}

$apiKeyPreview = $env:RUNPOD_API_KEY.Substring(0, [Math]::Min(8, $env:RUNPOD_API_KEY.Length))
Write-Host "🔑 RunPod API key found: $apiKeyPreview..." -ForegroundColor Green

# Function to create serverless endpoint
function Create-Endpoint {
    param(
        [string]$Name,
        [string]$Model,
        [string]$Gpu,
        [int]$MaxWorkers
    )
    
    Write-Host "📡 Creating $Name endpoint..." -ForegroundColor Blue
    
    $body = @{
        name = $Name
        template = @{
            imageName = "vllm/vllm-openai:latest"
            containerConfig = @{
                arguments = @(
                    "--model=$Model",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--trust-remote-code"
                )
                ports = @("8000/http")
            }
            volumeInGb = 50
            containerDiskInGb = 20
            env = @(
                @{key = "HF_HOME"; value = "/workspace/cache"}
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
        $response = Invoke-RestMethod -Uri "https://api.runpod.ai/v2/endpoints" `
            -Method POST `
            -Headers @{
                "Authorization" = "Bearer $env:RUNPOD_API_KEY"
                "Content-Type" = "application/json"
            } `
            -Body $body
        
        if ($response.id) {
            Write-Host "✅ $Name endpoint created: $($response.id)" -ForegroundColor Green
            return $response.id
        } else {
            Write-Host "❌ Failed to create $Name endpoint" -ForegroundColor Red
            Write-Host "Response: $response" -ForegroundColor Red
            return $null
        }
    }
    catch {
        Write-Host "❌ Error creating $Name endpoint: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Create Llama 8B endpoint
Write-Host ""
$llamaEndpoint8B = Create-Endpoint -Name "wisteria-llama-8b" -Model "meta-llama/Llama-3.1-8B-Instruct" -Gpu "NVIDIA_RTX4090" -MaxWorkers 3

# Create Llama 70B endpoint
Write-Host ""
$llamaEndpoint70B = Create-Endpoint -Name "wisteria-llama-70b" -Model "meta-llama/Llama-3.1-70B-Instruct" -Gpu "NVIDIA_A100_PCIE_40GB" -MaxWorkers 1

# Set environment variables
Write-Host ""
Write-Host "📝 Setting environment variables..." -ForegroundColor Blue

$env:RUNPOD_LLAMA_8B_ENDPOINT = $llamaEndpoint8B
$env:RUNPOD_LLAMA_70B_ENDPOINT = $llamaEndpoint70B

# Create configuration file
$configContent = @"
# RunPod Configuration for Wisteria CTR Studio
# Generated on $(Get-Date)

`$env:RUNPOD_API_KEY="$env:RUNPOD_API_KEY"
`$env:RUNPOD_LLAMA_8B_ENDPOINT="$llamaEndpoint8B"
`$env:RUNPOD_LLAMA_70B_ENDPOINT="$llamaEndpoint70B"

Write-Host "🚀 RunPod environment configured!" -ForegroundColor Green
Write-Host "Test with: python demo.py --provider runpod --runpod-model llama-8b --ad 'Test ad' --population-size 100" -ForegroundColor Yellow
"@

$configContent | Out-File -FilePath "runpod-config.ps1" -Encoding UTF8

Write-Host ""
Write-Host "🎉 RunPod setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Configuration saved to: runpod-config.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔧 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Load configuration: . .\runpod-config.ps1"
Write-Host ""
Write-Host "   2. Test the setup:"
Write-Host "      python demo.py --provider runpod --runpod-model llama-8b --ad 'Test ad' --population-size 100"
Write-Host ""
Write-Host "💰 Expected costs (when active):" -ForegroundColor Green
Write-Host "   - Llama 8B (RTX4090): `$0.39/hour"
Write-Host "   - Llama 70B (A100): `$1.89/hour"
Write-Host "   - Idle cost: `$0 (auto-scales to zero)"
Write-Host ""
Write-Host "⚠️  Note: Endpoints may take 5-10 minutes to initialize on first use" -ForegroundColor Yellow
Write-Host "    Subsequent requests will be much faster (30-90 seconds)"