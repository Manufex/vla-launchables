set dotenv-load := true
COMPOSE := "docker compose"
project := env_var_or_default("PROJECT", "lerobot-launchable")
config := env_var_or_default("CONFIG", "configs/pi05_default.yaml")

build:
  cd {{project}} && {{COMPOSE}} build

up:
  cd {{project}} && {{COMPOSE}} up -d

down:
  cd {{project}} && {{COMPOSE}} down

shell:
  cd {{project}} && {{COMPOSE}} run --rm lerobot bash

train:
  cd {{project}} && ./run.sh {{config}}

# Convenience commands for each model
train-pi:
  cd {{project}} && ./run.sh configs/pi05_default.yaml

train-smolvla:
  cd {{project}} && ./run.sh configs/smolvla_default.yaml

train-xvla:
  cd {{project}} && ./run.sh configs/xvla_default.yaml

train-groot:
  cd {{project}} && ./run.sh configs/groot_default.yaml

train-act:
  cd {{project}} && ./run.sh configs/act_default.yaml

