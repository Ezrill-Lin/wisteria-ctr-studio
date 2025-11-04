#!/bin/bash

echo "🚀 Setting up RunPod vLLM endpoints for Wisteria CTR Studio"
echo "============================================================"

# Check for API key
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "❌ RUNPOD_API_KEY environment variable is not set"
    echo ""
    echo "📋 Setup steps:"
    echo "1. Go to https://runpod.io/console/user/settings"
    echo "2. Generate an API key"
    echo "3. Run: export RUNPOD_API_KEY='your-api-key-here'"
    echo "4. Re-run this script"
    echo ""
    exit 1
fi

echo "🔑 RunPod API key found: ${RUNPOD_API_KEY:0:8}..."

# Function to create serverless endpoint
create_endpoint() {
    local name=$1
    local model=$2
    local gpu=$3
    local max_workers=$4
    
    echo "📡 Creating $name endpoint..."
    
    # Create endpoint using RunPod API
    response=$(curl -s -X POST "https://api.runpod.ai/v2/endpoints" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$name\",
            \"template\": {
                \"imageName\": \"vllm/vllm-openai:latest\",
                \"containerConfig\": {
                    \"arguments\": [
                        \"--model=$model\",
                        \"--host=0.0.0.0\",
                        \"--port=8000\",
                        \"--trust-remote-code\"
                    ],
                    \"ports\": [\"8000/http\"]
                },
                \"volumeInGb\": 50,
                \"containerDiskInGb\": 20,
                \"env\": [
                    {\"key\": \"HF_HOME\", \"value\": \"/workspace/cache\"}
                ]
            },
            \"workersMin\": 0,
            \"workersMax\": $max_workers,
            \"gpuTypeId\": \"$gpu\",
            \"scalerType\": \"QUEUE_DELAY\",
            \"scalerSettings\": {
                \"queueDelay\": 5,
                \"maxIdleTime\": 300
            }
        }")
    
    endpoint_id=$(echo $response | jq -r '.id // empty')
    
    if [ -n "$endpoint_id" ] && [ "$endpoint_id" != "null" ]; then
        echo "✅ $name endpoint created: $endpoint_id"
        echo "$endpoint_id"
    else
        echo "❌ Failed to create $name endpoint"
        echo "Response: $response"
        echo ""
    fi
}

# Create Llama 8B endpoint
echo ""
LLAMA_8B_ENDPOINT=$(create_endpoint \
    "wisteria-llama-8b" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "NVIDIA_RTX4090" \
    3)

# Create Llama 70B endpoint  
echo ""
LLAMA_70B_ENDPOINT=$(create_endpoint \
    "wisteria-llama-70b" \
    "meta-llama/Llama-3.1-70B-Instruct" \
    "NVIDIA_A100_PCIE_40GB" \
    1)

# Create environment configuration file
echo ""
echo "📝 Creating environment configuration..."

cat > runpod-config.env << EOF
# RunPod Configuration for Wisteria CTR Studio
# Generated on $(date)

export RUNPOD_API_KEY="$RUNPOD_API_KEY"
export RUNPOD_LLAMA_8B_ENDPOINT="$LLAMA_8B_ENDPOINT"
export RUNPOD_LLAMA_70B_ENDPOINT="$LLAMA_70B_ENDPOINT"

# Usage:
# source runpod-config.env
# python demo.py --provider runpod --runpod-model llama-8b --ad "Test ad" --population-size 100
EOF

# Create PowerShell version for Windows
cat > runpod-config.ps1 << EOF
# RunPod Configuration for Wisteria CTR Studio (PowerShell)
# Generated on $(date)

\$env:RUNPOD_API_KEY="$RUNPOD_API_KEY"
\$env:RUNPOD_LLAMA_8B_ENDPOINT="$LLAMA_8B_ENDPOINT"
\$env:RUNPOD_LLAMA_70B_ENDPOINT="$LLAMA_70B_ENDPOINT"

Write-Host "🚀 RunPod environment configured!"
Write-Host "Test with: python demo.py --provider runpod --runpod-model llama-8b --ad 'Test ad' --population-size 100"
EOF

echo ""
echo "🎉 RunPod setup complete!"
echo ""
echo "📋 Configuration saved to:"
echo "   - runpod-config.env (Linux/Mac)"
echo "   - runpod-config.ps1 (Windows)"
echo ""
echo "🔧 Next steps:"
echo "   1. Load configuration:"
echo "      Linux/Mac: source deploy/runpod/runpod-config.env"
echo "      Windows:   . deploy/runpod/runpod-config.ps1"
echo ""
echo "   2. Test the setup:"
echo "      python demo.py --provider runpod --runpod-model llama-8b --ad 'Test ad' --population-size 100"
echo ""
echo "💰 Expected costs (when active):"
echo "   - Llama 8B (RTX4090): \$0.39/hour"
echo "   - Llama 70B (A100): \$1.89/hour"
echo "   - Idle cost: \$0 (auto-scales to zero)"
echo ""
echo "⚠️  Note: Endpoints may take 5-10 minutes to initialize on first use"
echo "    Subsequent requests will be much faster (30-90 seconds)"