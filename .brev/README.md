# Brev Launchable Configuration

This directory contains the configuration files required to launch the Spelling Bee Assistant on Brev.

## Files

- **setup.sh** - Automated setup script that installs dependencies and prepares the environment
- **validate.py** - Configuration validator to check if your setup is ready for Brev
- **QUICKSTART.md** - Step-by-step guide for launching with Brev
- **launch-example.yaml** - Example configuration for customizing your deployment
- **(parent) .brev.yaml** - Main Brev configuration file with app metadata and requirements

## What is Brev?

Brev is NVIDIA's platform for easily deploying and launching AI applications. It provides a simple way to run GPU-accelerated workloads with minimal configuration.

## Configuration Overview

The Brev configuration for this application includes:

- **Runtime**: Python 3.12+
- **Port**: 8080 (HTTP/WebSocket)
- **Dependencies**: Redis (for session storage)
- **Required Secrets**: ElevenLabs API key
- **Optional Services**: vLLM model endpoint

## Setup Process

When launched via Brev, the following happens automatically:

1. Python 3.12+ environment is provisioned
2. System dependencies (tesseract-ocr, build tools) are installed
3. Python packages are installed from requirements.txt
4. NVIDIA Pipecat compatibility patches are applied
5. Redis service is started for session storage
6. Application starts on port 8080

## Required Configuration

Before launching, ensure you have:

1. **ElevenLabs API Key** - Required for speech-to-text and text-to-speech
   - Set as `ELEVENLABS_API_KEY` environment variable
   - Get your key from: https://elevenlabs.io/

2. **Optional: vLLM Endpoint** - For vision-language model
   - Default: expects service at `http://vllm-nemotron-nano-vl-8b:5566/v1`
   - Can be overridden with `VLLM_VL_BASE` environment variable

## Manual Setup (Development)

If you want to test the setup locally:

```bash
# Validate your configuration first
python3 .brev/validate.py

# Run the setup script
bash .brev/setup.sh

# Set required environment variables
export ELEVENLABS_API_KEY=your-key-here

# Start Redis (if not using Brev's managed service)
redis-server &

# Start the application
python spelling_bee_agent_backend.py
```

## Troubleshooting

### Import Errors

If you see pipecat-related import errors, the patches may not have been applied. Run:

```bash
python patch_nvidia_pipecat.py $(python -c "import nvidia_pipecat; import os; print(os.path.dirname(nvidia_pipecat.__file__))")
```

### Redis Connection Issues

Ensure Redis is running and accessible at the URL specified in `REDIS_URL` (default: `redis://localhost:6379/0`).

### Missing ElevenLabs API Key

The application cannot start voice sessions without a valid ElevenLabs API key. Set it as:

```bash
export ELEVENLABS_API_KEY=your-key-here
```

## Support

For issues specific to Brev deployment, consult the [Brev documentation](https://docs.nvidia.com/brev/).

For application-specific issues, see the main [README.md](../README.md) or file an issue on GitHub.
