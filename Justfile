set dotenv-load := true
COMPOSE := "docker compose"
project := env_var_or_default("PROJECT", "lerobot-pi05")

build:
  cd {{project}} && {{COMPOSE}} build

up:
  cd {{project}} && {{COMPOSE}} up -d

down:
  cd {{project}} && {{COMPOSE}} down

shell:
  cd {{project}} && {{COMPOSE}} run --rm pi05 bash

train:
  cd {{project}} && ./train.sh

