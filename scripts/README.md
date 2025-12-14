# Startup Scripts

These scripts are designed to be used as **Brev launchable setup scripts**. They build the Docker container and provide instructions for running training.

## Available Scripts

- `startup_pi05.sh` - LeRobot π0.5
- `startup_smolvla.sh` - SmolVLA
- `startup_xvla.sh` - X-VLA
- `startup_groot.sh` - GR00T-N1.5
- `startup_act.sh` - ACT

## Usage

### For Brev Launchables

1. When creating a launchable on Brev, use the appropriate startup script as the setup script
2. The script will:
   - Build the Docker container automatically
   - Set up the environment
   - Display instructions for running training

### Example: Creating a Brev Launchable

1. Go to Brev dashboard → Create Launchable
2. In the setup script section, paste the contents of `startup_pi05.sh` (or the model you want)
3. The instance will build the container on startup
4. After SSH'ing in, users can immediately run training

## What Each Script Does

1. **Sets up paths**: Detects the project directory (works in both `$HOME` and `/home/ubuntu`)
2. **Builds container**: Runs `docker compose build` to prepare the environment
3. **Provides instructions**: Shows users how to run training after SSH'ing in

## After SSH'ing In

Users can run training with:

```bash
# Direct method
./run.sh configs/pi05_default.yaml

# Or using just
just train-pi
```

The training script will prompt for:
- `HF_TOKEN` - Hugging Face token
- `WANDB_API_KEY` - Weights & Biases API key

## Customization

Users can:
- Edit the config files in `lerobot-launchable/configs/` to customize training
- Create custom configs: `cp configs/pi05_default.yaml configs/pi05_custom.yaml`
- Override via environment variables (though config files are preferred)

