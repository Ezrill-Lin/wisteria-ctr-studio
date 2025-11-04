#!/bin/bash
# Setup GKE cluster with GPU support for vLLM deployment

set -e

PROJECT_ID=${PROJECT_ID:-"your-project-id"}
CLUSTER_NAME=${CLUSTER_NAME:-"vllm-cluster"}
ZONE=${ZONE:-"us-central1-a"}
REGION=${REGION:-"us-central1"}

echo "🚀 Setting up GKE cluster for vLLM deployment..."
echo "Project: $PROJECT_ID"
echo "Cluster: $CLUSTER_NAME"
echo "Zone: $ZONE"

# Enable required APIs
echo "📡 Enabling required GCP APIs..."
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create the main cluster (without GPUs initially)
echo "🏗️ Creating GKE cluster..."
gcloud container clusters create $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=e2-standard-4 \
    --num-nodes=1 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5 \
    --enable-autorepair \
    --enable-autoupgrade \
    --enable-ip-alias \
    --network="default" \
    --subnetwork="default"

# Add GPU node pool for T4 (Llama 8B)
echo "🎮 Adding T4 GPU node pool for Llama 8B..."
gcloud container node-pools create t4-pool \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=0 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=5 \
    --enable-autorepair \
    --preemptible \
    --node-labels=gpu-type=t4,model-support=llama-8b \
    --node-taints=nvidia.com/gpu=present:NoSchedule

# Add GPU node pool for A100 (Llama 70B)
echo "🚀 Adding A100 GPU node pool for Llama 70B..."
gcloud container node-pools create a100-pool \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --machine-type=a2-highgpu-2g \
    --accelerator=type=nvidia-tesla-a100,count=2 \
    --num-nodes=0 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=2 \
    --enable-autorepair \
    --preemptible \
    --node-labels=gpu-type=a100,model-support=llama-70b \
    --node-taints=nvidia.com/gpu=present:NoSchedule

# Get cluster credentials
echo "🔑 Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE --project=$PROJECT_ID

# Install NVIDIA GPU drivers
echo "🔧 Installing NVIDIA GPU drivers..."
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Create namespace for vLLM
echo "📦 Creating vllm namespace..."
kubectl create namespace vllm --dry-run=client -o yaml | kubectl apply -f -

echo "✅ GKE cluster setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Wait for GPU drivers to install (2-3 minutes)"
echo "2. Deploy vLLM services: kubectl apply -f ."
echo "3. Check status: kubectl get pods -n vllm"
echo ""
echo "🔍 Useful commands:"
echo "  kubectl get nodes -o wide"
echo "  kubectl describe nodes"
echo "  kubectl get pods -n vllm -o wide"