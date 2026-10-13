# Adaptive-Group Process Rewards for GUI Agents

[![Website](https://img.shields.io/badge/Website-AGPR-2663ff)](https://shzirui.github.io/AGPR/)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/shzirui/AGPR)

🌐 [Website](https://shzirui.github.io/AGPR/) | 💻 [Code](https://github.com/shzirui/AGPR)

This repository extends the [DART-GUI](https://github.com/computer-use-agents/dart-gui) training infrastructure with **adaptive-group process rewards**. A completed GUI trajectory is partitioned into subgoal groups. A critic scores each action and each group, and the resulting process rewards shape GRPO's step-level advantage.

## Process rewards in training

Process rewards are produced while rollouts are collected, then consumed by the asynchronous trainer.

* Rollout scoring lives in `dart_rollouter/src/core/trajectory_runner.py`, with grouping and the two-level critic in `dart_rollouter/src/core/`.
* The trainer reads those scores in `verl/trainer/ppo/ray_trainer_async.py`.
* The training entry is `examples/osworld/async/debug.sh`.

`use_pr=3` is the setting used in the paper: keep GRPO's outcome advantage and add a process advantage standardized over the sampled group.

```bash
use_pr=3
pr_adv_weight=0.5          # alpha in the paper
pr_multiply_scores=true    # action score x refined group score
```

Set `use_pr=none` for outcome-only GRPO. `pr_adv_weight` is the coefficient on the process advantage.

## Quick start

The pipeline follows DART-GUI: prepare containers and MySQL, start the rollout model service, start the GUI runner, then start training. Paths, hosts, and passwords are environment variables or placeholders. Set them for your machine before launching.

For the original DART-GUI image, SQL schema, and GUI environment, see:

* DART-GUI: https://github.com/computer-use-agents/dart-gui
* GUI environment: `GUI-Docker-Env/`

### 1. Preparation

Download a DART-GUI-compatible rollout image, trainer image, and GUI environment image for your accelerator. Also pull MySQL:

```bash
docker pull mysql:8.0.44-debian
```

Download the base GUI agent and the process-reward critic separately. The paper uses UI-TARS-1.5-7B for the agent and Qwen3.6-27B for the critic.

```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir /path/to/UI-TARS-1.5-7B
```

Point `MODEL_PATH` and `dart_rollouter/config/config.yaml` (`model.ckpt_path`) at that checkpoint.

### 2. Docker

Start a rollout container, a trainer container, and MySQL. Mount this repository at the same path in the rollout and trainer containers, for example `/path/to/AGPR`.

```bash
docker run -dit \
  --name mysql-server \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD}" \
  -v /path/to/mysql:/var/lib/mysql \
  mysql:8.0.44-debian
```

```bash
export DB_HOST="127.0.0.1"
export DB_USER="root"
export DB_PASSWORD="${DB_PASSWORD}"
export DB_DATABASE="dart"
export DB_PORT="3306"
export DB_CHARSET="utf8mb4"
```

Put the same host, user, password, and database in `dart_rollouter/config/config.yaml` under `mysql`. Use the DART-GUI schema for `rollout_run` and `checkpoint`.

### 3. Rollout

Inside the rollout container, edit `dart_rollouter/config/config.yaml`:

* `model.ckpt_path`
* `model.service_port` and `model.service_endpoint`
* `storage.root`, which must match `ROLLOUTER_DATA_DIR` / `RUN_ID` used by the trainer
* `mysql.*`

Then start the model service and the agent runner:

```bash
cd dart_rollouter
bash model_service.sh
bash run.sh
```

Completed trajectories are written to MySQL, including the process-reward column consumed by training.

### 4. Training

Inside the trainer container, from the repository root:

```bash
export MODEL_PATH=/path/to/UI-TARS-1.5-7B
export ROLLOUTER_DATA_DIR=dart_rollouter/results/osworld_rollout
export RUN_ID=results/osworld_rollout
export ROLLOUT_SERVER_URL=http://127.0.0.1:15959
export DB_PASSWORD=your_mysql_password
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4

bash examples/osworld/async/debug.sh
```

Other fields to check in `examples/osworld/async/debug.sh` are `N_NODES`, batch sizes, and `use_pr`. Optional SwanLab logging is used only when `SWANLAB_API_KEY` is set.

## Acknowledgments

This repository is built on [DART-GUI](https://github.com/computer-use-agents/dart-gui) and [verl](https://github.com/volcengine/verl), and uses vLLM for rollout serving.

## Citation

The citation will be added after the arXiv version is posted.
