# RunPod GitHub Deployment Guide

This directory contains ready-to-use files for deploying Llama models on RunPod using GitHub integration.

## 🚀 Quick Setup (GitHub Integration)

### Step 1: Ensure Your Repository is Public
Make sure your `Wisteria-CTR-Studio` repository is public on GitHub so RunPod can access it.

### Step 2: Create Serverless Endpoints

#### For Llama 8B:
1. Go to [RunPod Serverless Console](https://runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Select **"Create Custom Template"**
4. Click **"Import from GitHub"**
5. Enter:
   - **Repository**: `Ezrill-Lin/Wisteria-CTR-Studio`
   - **Branch**: `main`
   - **Config File**: `deploy/runpod/runpod-template-8b.json`
6. Click **"Import Template"**
7. Configure the endpoint:
   - **Name**: `wisteria-llama-8b`
   - **GPU**: RTX 4090 (or RTX 3090, A4000, A5000)
   - **Min Workers**: 0
   - **Max Workers**: 3
8. Click **"Create Endpoint"**

#### For Llama 70B:
1. Create another endpoint using:
   - **Repository**: `Ezrill-Lin/Wisteria-CTR-Studio`
   - **Config File**: `deploy/runpod/runpod-template-70b.json`
   - **Name**: `wisteria-llama-70b`
   - **GPU**: A100 40GB (or H100)
   - **Min Workers**: 0
   - **Max Workers**: 1

### Step 3: Set Environment Variables
After creating endpoints, copy their IDs:

```powershell
$env:RUNPOD_LLAMA_8B_ENDPOINT="your-8b-endpoint-id"
$env:RUNPOD_LLAMA_70B_ENDPOINT="your-70b-endpoint-id"
```

### Step 4: Test Your Setup
```bash
cd ../..
python demo.py --provider runpod --runpod-model llama-8b --ad "Premium coffee" --population-size 10
```

## 📁 Files Created

- **`Dockerfile`**: Custom Docker image based on vLLM
- **`start-llama-8b.sh`**: Startup script for 8B model
- **`start-llama-70b.sh`**: Startup script for 70B model
- **`runpod-template-8b.json`**: RunPod template for 8B model
- **`runpod-template-70b.json`**: RunPod template for 70B model

## 🔧 Technical Details

### Model Configuration:
- **8B Model**: 16GB download, ~80GB volume, RTX 4090 compatible
- **70B Model**: 140GB download, ~150GB volume, requires A100+

### Optimizations:
- Persistent model caching on `/runpod-volume`
- GPU memory optimization (90-95% utilization)
- Automatic tensor parallelism for 70B model
- OpenAI-compatible API endpoint

### Costs:
- **8B on RTX 4090**: ~$0.39/hour when active
- **70B on A100**: ~$1.89/hour when active
- **Idle cost**: $0 (auto-scales to zero)

## 🛠️ Alternative: Manual Template Import

If GitHub import doesn't work, you can manually upload the template files:

1. Copy the contents of `runpod-template-8b.json`
2. In RunPod console, click "Create Custom Template"
3. Paste the JSON configuration
4. Adjust repository URL if needed

## 🔍 Troubleshooting

### Common Issues:
1. **Repository not accessible**: Ensure repo is public
2. **Template import fails**: Check JSON syntax
3. **Model download timeout**: Increase idle timeout in template
4. **GPU out of memory**: Reduce `gpu-memory-utilization` in startup scripts

### Monitoring:
- Check endpoint logs in RunPod console
- Monitor costs at: https://runpod.io/console/billing
- First deployment takes 5-10 minutes (model download)
- Subsequent starts: 30-90 seconds

## 📈 Benefits of GitHub Integration

✅ **Version Control**: Automatic updates when you push changes  
✅ **Reproducible**: Exact same deployment every time  
✅ **No Manual Steps**: Complete automation  
✅ **Easy Rollbacks**: Use different Git branches/tags  
✅ **Team Collaboration**: Share templates via Git