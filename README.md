# VLA Launchables

**The easiest way to fine-tune Vision-Language-Action (VLA) models.** No setup, no configuration headaches—just SSH into a launchable, enter your credentials, specify your dataset/training parameters and start training.

## Why VLA Launchables?

Fine-tuning VLA models has never been easier:
- **One-click deployment** on NVIDIA Brev
- **Zero configuration**—just add your training parameters
- **Automatic dataset fetching** from Hugging Face
- **Real-time training tracking** on Weights & Biases
- **Automatic model upload** to Hugging Face (private by default)
- **Auto-delete instances** to save costs (optional)

## Quick Start

### Deploy on Brev

Click one of the deploy buttons below to spin up an instance:

**LeRobot π0.5**
[![Deploy on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-36lFUACCCYMoYRnJQO2WISupFDU)

**SmolVLA**
[![Deploy on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=YOUR_SMOLVLA_LAUNCHABLE_ID)

**GR00T-N1.5**
[![Deploy on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-36q0gIw6PpTQlakNOTBPrfgDM8d)

**X-VLA**
[![Deploy on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=YOUR_XVLA_LAUNCHABLE_ID)

**ACT**
[![Deploy on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=YOUR_ACT_LAUNCHABLE_ID)

### After SSH'ing In

1. **Enter your credentials** (prompted automatically):
   - Hugging Face token (`HF_TOKEN`)
   - Weights & Biases API key (`WANDB_API_KEY`)
   - Brev token (`BREV_TOKEN`) - optional, only if you want auto-delete

2. **Customize training**:
   ```bash
   nano configs/pi05_default.yaml
   # Edit: dataset.repo_id, steps, batch_size, output_dir, etc.
   ```

3. **Start training**:
   ```bash
   ./run.sh configs/pi05_default.yaml
   # or
   just train-pi
   ```

That's it! The script will:
- Fetch your dataset from Hugging Face
- Train the model with your configuration
- Track progress on Weights & Biases
- Upload the trained model to Hugging Face (private)
- Optionally delete the Brev instance to save costs

## What You Need

- **Hugging Face token**: [Get one here](https://huggingface.co/settings/tokens)
- **Weights & Biases API key**: [Get one here](https://wandb.ai/authorize)
- **Brev token** (optional): Only needed if you enable auto-delete in the config

> [!IMPORTANT]
> Credentials are never stored in config files. They're either set as environment variables or prompted interactively.


## Local Usage

You can also run this locally on your own GPU machine:

```bash
# Build the container
cd lerobot-launchable
docker compose build

# Run training
./run.sh configs/pi05_default.yaml
```

**Requirements:**
- Linux with NVIDIA GPU
- Docker + Docker Compose v2

## Creating Your Own Launchable

If you'd like to create a custom Brev launchable for one of these models (or your own), you can fork this repo and configure it on Brev.

### Configuring a custom Brev launchable

1. Log in to the [Brev](https://login.brev.nvidia.com/signin) website.
2. Go to the Launchables category.
3. Click the **Create Launchable** button.
4. Choose the "I have code files in a git repository" option.
5. Insert "https://github.com/Manufex/vla-launchables" or your own forked version
6. Choose **VM Mode - Basic VM with Python installed**, then click Next.
7. On the next page, add a setup script. Under the *Paste Script* tab, add one of the startup scripts from `scripts/` or your own customized version
8. Click Next.
9. Under "Do you want a Jupyter Notebook experience" select "No, I don't want Jupyter".
10. Click Next.
11. Choose your desired compute (GPU required for training).
12. Choose disk storage, then click Next.
13. Enter a name, then select **Create Launchable**.

## Troubleshooting

**CUDA out of memory?**
- Lower `batch_size` in your config file (try 4 or 2)

**Training failed?**
- Check your `HF_TOKEN` and `WANDB_API_KEY` are correct
- Verify your dataset `repo_id` exists on Hugging Face

## License

MIT license
