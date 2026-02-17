# Quick Start Guide: Launching with Brev

This guide will help you launch the Spelling Bee Assistant on Brev in just a few minutes.

## Prerequisites

1. **Brev Account**: Sign up at [brev.dev](https://brev.dev) if you don't have an account
2. **Brev CLI** (optional, for command-line deployment):
   ```bash
   # Install Brev CLI
   curl -fsSL https://brev.dev/install | bash
   
   # Login to your account
   brev login
   ```
3. **ElevenLabs API Key**: Get your free API key from [elevenlabs.io](https://elevenlabs.io)

## Method 1: Launch via Brev Dashboard (Recommended)

1. **Navigate to Brev Dashboard**
   - Go to your Brev dashboard at [brev.dev/dashboard](https://brev.dev/dashboard)

2. **Create New Launch**
   - Click "New Launch" or "Deploy Application"
   - Select "From GitHub Repository"
   - Enter repository: `https://github.com/elizabetht/spelling-bee-assistant`

3. **Configure Environment**
   - Add required secret: `ELEVENLABS_API_KEY` with your API key
   - (Optional) Configure other environment variables from `.brev/launch-example.yaml`

4. **Deploy**
   - Click "Launch" or "Deploy"
   - Wait for deployment to complete (typically 3-5 minutes)
   - Access your application at the provided URL

## Method 2: Launch via Brev CLI

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/elizabetht/spelling-bee-assistant.git
   cd spelling-bee-assistant
   ```

2. **Configure secrets**:
   ```bash
   # Set your ElevenLabs API key
   brev secret set ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
   ```

3. **Launch the application**:
   ```bash
   brev launch
   ```
   
   The CLI will:
   - Read the `.brev.yaml` configuration
   - Provision resources (CPU, memory, Redis)
   - Install dependencies via `.brev/setup.sh`
   - Start the application
   - Provide you with a URL to access the app

4. **Access your application**:
   ```bash
   brev open
   ```
   This will open your application URL in your default browser.

## Post-Deployment

### Accessing the Application

Once deployed, you'll receive a URL like: `https://spelling-bee-assistant-xyz.brev.dev`

1. Open the URL in your browser
2. You'll see the Spelling Bee Assistant UI
3. Upload a word list image (photo of spelling words)
4. Click "Start Session" to begin the voice-guided spelling practice

### Managing Your Deployment

```bash
# View logs
brev logs

# Check status
brev status

# Update environment variables
brev secret set VARIABLE_NAME=value

# Restart application
brev restart

# Stop application
brev stop

# Delete deployment
brev delete
```

## Optional: Custom vLLM Endpoint

If you're hosting your own vLLM model service:

```bash
# Set custom vLLM endpoint
brev env set VLLM_VL_BASE=https://your-vllm-service.com/v1
brev env set VLLM_VL_MODEL=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8

# Restart to apply changes
brev restart
```

## Optional: Enable NeMo Guardrails

For enhanced content safety (recommended for production):

```bash
brev env set ENABLE_NEMO_GUARDRAILS=true
brev restart
```

## Troubleshooting

### Deployment Fails

1. Check logs: `brev logs`
2. Verify your ElevenLabs API key is valid
3. Ensure Python 3.12+ is specified in runtime configuration

### Application Won't Start

1. Check if Redis service is running: `brev services status`
2. Verify all required environment variables are set: `brev env list`
3. Review setup logs for dependency installation errors

### Voice Session Issues

1. Ensure `ELEVENLABS_API_KEY` is correctly set and valid
2. Check that WebSocket connections are allowed (should be automatic on Brev)
3. Try a different browser if audio issues persist

### Can't Access the UI

1. Verify the application is running: `brev status`
2. Check the health endpoint: `curl https://your-app.brev.dev/healthz`
3. Review application logs: `brev logs`

## Support

- **Brev Platform Issues**: [Brev Documentation](https://docs.nvidia.com/brev/) or Brev support
- **Application Issues**: [GitHub Issues](https://github.com/elizabetht/spelling-bee-assistant/issues)
- **Main Documentation**: [README.md](../README.md)

## Cost Considerations

- **Brev Resources**: Check current Brev pricing for compute resources
- **ElevenLabs API**: Free tier available, check usage limits
- **Redis**: Included in Brev managed services (no additional cost)

## Next Steps

Once your application is running:

1. Try uploading different word list images
2. Test the voice interaction features
3. Explore the NeMo Guardrails configuration in `./guardrails/`
4. Consider deploying your own vLLM model for complete self-hosting

---

**Happy Spelling Practice! 📚✨**
