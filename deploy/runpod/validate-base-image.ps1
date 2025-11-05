# =========================================================
# Validate RunPod vLLM Base Image
# Quick check before building to avoid wasting hours
# =========================================================

param(
    [string]$BaseImage = "runpod/worker-v1-vllm:v2.9.6"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RunPod vLLM Base Image Validator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: Docker is running
Write-Host "[1/6] Checking Docker..." -ForegroundColor Yellow
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Docker is running" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Docker is not running!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ✗ Docker is not installed or not running!" -ForegroundColor Red
    exit 1
}

# Check 2: Pull the base image
Write-Host "[2/6] Pulling base image: $BaseImage..." -ForegroundColor Yellow
docker pull $BaseImage
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to pull base image!" -ForegroundColor Red
    Write-Host "  The image name might be incorrect or not exist on Docker Hub" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Base image pulled successfully" -ForegroundColor Green

# Check 3: Inspect the image
Write-Host "[3/6] Inspecting image..." -ForegroundColor Yellow
$imageInfo = docker inspect $BaseImage | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to inspect image!" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Image exists and is valid" -ForegroundColor Green
Write-Host "    Size: $([math]::Round($imageInfo[0].Size / 1GB, 2)) GB" -ForegroundColor Gray

# Check 4: Verify required tools exist in image
Write-Host "[4/6] Checking for required tools..." -ForegroundColor Yellow

$requiredCommands = @(
    "python3",
    "pip",
    "huggingface-cli"
)

$missingCommands = @()

foreach ($cmd in $requiredCommands) {
    $result = docker run --rm $BaseImage which $cmd 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Found: $cmd" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $cmd" -ForegroundColor Red
        $missingCommands += $cmd
    }
}

if ($missingCommands.Count -gt 0) {
    Write-Host ""
    Write-Host "  ✗ Missing required commands: $($missingCommands -join ', ')" -ForegroundColor Red
    Write-Host "  This base image may not work for model pre-download!" -ForegroundColor Red
    exit 1
}

# Check 5: Verify Python packages
Write-Host "[5/6] Checking Python packages..." -ForegroundColor Yellow

$requiredPackages = @(
    "vllm",
    "transformers",
    "huggingface_hub"
)

$missingPackages = @()

foreach ($pkg in $requiredPackages) {
    $result = docker run --rm $BaseImage python3 -c "import $($pkg.Replace('_', '-').Replace('huggingface-hub', 'huggingface_hub'))" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Found: $pkg" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $pkg" -ForegroundColor Red
        $missingPackages += $pkg
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "  ✗ Missing required packages: $($missingPackages -join ', ')" -ForegroundColor Red
    Write-Host "  This base image may not work!" -ForegroundColor Red
    exit 1
}

# Check 6: Test huggingface-cli download capability
Write-Host "[6/6] Testing HuggingFace download capability..." -ForegroundColor Yellow
$testResult = docker run --rm $BaseImage python3 -c "from huggingface_hub import snapshot_download; print('OK')" 2>&1
if ($testResult -match "OK") {
    Write-Host "  ✓ HuggingFace download functions work" -ForegroundColor Green
} else {
    Write-Host "  ✗ HuggingFace download test failed!" -ForegroundColor Red
    Write-Host "  Error: $testResult" -ForegroundColor Red
    exit 1
}

# Check 7: Environment variable check
Write-Host ""
Write-Host "Additional Information:" -ForegroundColor Cyan
Write-Host "  Base Image: $BaseImage" -ForegroundColor Gray
Write-Host "  Architecture: $($imageInfo[0].Architecture)" -ForegroundColor Gray
Write-Host "  OS: $($imageInfo[0].Os)" -ForegroundColor Gray

# Check HF_TOKEN
Write-Host ""
if ($env:HF_TOKEN) {
    Write-Host "  ✓ HF_TOKEN environment variable is set" -ForegroundColor Green
} else {
    Write-Host "  ⚠ HF_TOKEN environment variable is NOT set" -ForegroundColor Yellow
    Write-Host "    Set it with: `$env:HF_TOKEN = 'your_token_here'" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ All checks passed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The base image is correctly configured." -ForegroundColor Green
Write-Host "You can proceed with building your custom image." -ForegroundColor Green
Write-Host ""
Write-Host "Next step:" -ForegroundColor Cyan
Write-Host "  .\build-8b.ps1" -ForegroundColor Yellow
Write-Host ""
