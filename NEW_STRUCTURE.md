# New Simplified Structure

## Directory Layout

```
vla-launchables/
├── lerobot-launchable/          # Single launchable directory
│   ├── Dockerfile               # Base image (core lerobot)
│   ├── docker-compose.yml       # Single compose file
│   ├── run.sh                   # Training wrapper
│   ├── README.md                # Usage instructions
│   ├── scripts/
│   │   ├── run_train.py         # Python training script
│   │   └── prompt_tokens.sh     # Token prompts
│   └── configs/
│       ├── pi05_default.yaml
│       ├── smolvla_default.yaml
│       ├── xvla_default.yaml
│       ├── groot_default.yaml
│       └── act_default.yaml
├── outputs/                     # Training outputs
├── cache/                       # Cache directory
└── Justfile                     # Updated launcher
```

## Key Changes

1. **Single Docker container**: One `Dockerfile` and `docker-compose.yml` for all models
2. **Config-driven**: Model-specific settings in YAML files
3. **Runtime extras installation**: Policy extras installed based on `policy.type` in config
4. **Simplified structure**: No per-model directories needed

## Usage

```bash
# Build once
cd lerobot-launchable
just build

# Train any model
just train-pi
just train-smolvla
just train-xvla
just train-groot
just train-act

# Or with custom config
./run.sh configs/pi05_custom.yaml
```

## Migration Notes

- Old structure (`launchables/pi/`, etc.) can be removed
- Old `common/` directory can be removed (functionality moved to `lerobot-launchable/scripts/`)
- All models now use the same Docker image
- Config files replace per-model docker-compose files

