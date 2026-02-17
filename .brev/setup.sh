#!/usr/bin/env bash
# Brev setup script for Spelling Bee Assistant
# This script is automatically executed by Brev during deployment

set -euo pipefail

echo "=== Brev Setup: Spelling Bee Assistant ==="

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "ERROR: Python 3.12+ is required (found $PYTHON_VERSION)"
    exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ca-certificates \
    git \
    g++ \
    build-essential

# Clean up apt cache
rm -rf /var/lib/apt/lists/*

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Install and patch pipecat
echo "Installing and patching pipecat..."
pip install --force-reinstall --no-deps "pipecat-ai[elevenlabs]>=0.0.100" --quiet

# Apply nvidia-pipecat patches
echo "Applying nvidia-pipecat compatibility patches..."
NVPC=$(python -c "import nvidia_pipecat; import os; print(os.path.dirname(nvidia_pipecat.__file__))" 2>/dev/null || echo "")

if [[ -n "${NVPC}" ]] && [[ -d "${NVPC}" ]]; then
    # Patch 1: Fix import statement
    sed -i 's/from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType/from pipecat.serializers.base_serializer import FrameSerializer/' \
        "${NVPC}/transports/network/ace_fastapi_websocket.py" 2>/dev/null || true
    
    # Patch 2: Fix serializer type check
    sed -i 's/is_binary = self._params.serializer.type == FrameSerializerType.BINARY/is_binary = True/' \
        "${NVPC}/transports/network/ace_fastapi_websocket.py" 2>/dev/null || true
    
    # Patch 3: Apply custom patch script
    if [[ -f "patch_nvidia_pipecat.py" ]]; then
        python patch_nvidia_pipecat.py "${NVPC}" || true
    fi
    
    echo "Patches applied successfully"
else
    echo "WARNING: nvidia-pipecat not found or not installed, skipping patches"
fi

# Verify installation
echo "Verifying installation..."
python -c "import fastapi; print('✓ FastAPI installed')"
python -c "import uvicorn; print('✓ Uvicorn installed')"
python -c "import redis; print('✓ Redis client installed')"
python -c "import elevenlabs; print('✓ ElevenLabs SDK installed')"

# Check optional dependencies
python -c "import pipecat; print('✓ Pipecat installed')" 2>/dev/null || echo "⚠ Pipecat not available"
python -c "import nvidia_pipecat; print('✓ NVIDIA Pipecat installed')" 2>/dev/null || echo "⚠ NVIDIA Pipecat not available"
python -c "import nemoguardrails; print('✓ NeMo Guardrails installed')" 2>/dev/null || echo "⚠ NeMo Guardrails not available"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Required environment variables:"
echo "  - ELEVENLABS_API_KEY: Your ElevenLabs API key"
echo ""
echo "Optional environment variables:"
echo "  - REDIS_URL: Redis connection (default: redis://localhost:6379/0)"
echo "  - VLLM_VL_BASE: vLLM endpoint (default: http://vllm-nemotron-nano-vl-8b:5566/v1)"
echo "  - ENABLE_NEMO_GUARDRAILS: Enable guardrails (default: false)"
echo ""
echo "Ready to start with: python spelling_bee_agent_backend.py"
