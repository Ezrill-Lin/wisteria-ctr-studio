# =========================================================
# Build vLLM Docker Image with Llama 3.1 8B Instruct
# =========================================================

param(
    [string]$DockerUser = "ezrill",
    [string]$HfToken = $env:HF_TOKEN
)

# Check if HF_TOKEN is set
if ([string]::IsNullOrEmpty($HfToken)) {
    Write-Host "Error: HF_TOKEN environment variable not set" -ForegroundColor Red
    Write-Host "Set it with: `$env:HF_TOKEN = 'your_token_here'" -ForegroundColor Yellow
    exit 1
}

$ModelName = "meta-llama/Llama-3.1-8B-Instruct"
$ImageTag = "$DockerUser/vllm-llama-8b:latest"

Write-Host "Building Docker image for Llama 3.1 8B..." -ForegroundColor Cyan
Write-Host "Model: $ModelName" -ForegroundColor Gray
Write-Host "Image tag: $ImageTag" -ForegroundColor Gray
Write-Host ""

# Build the Docker image
docker build `
    --build-arg MODEL_NAME="$ModelName" `
    --build-arg HF_TOKEN="$HfToken" `
    -t $ImageTag `
    -f Dockerfile `
    .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Push to Docker Hub: docker push $ImageTag" -ForegroundColor Yellow
    Write-Host "2. Use in RunPod with image: $ImageTag" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
