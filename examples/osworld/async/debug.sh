#!/usr/bin/env bash
# Train a GUI agent with adaptive-group process rewards.
#
# Edit the variables in the "User configuration" block, then run from the
# repository root:
#   bash examples/osworld/async/debug.sh
#
# use_pr=true is the paper setting: GRPO's outcome advantage plus a standardized
# process advantage, mixed with pr_adv_weight (alpha in the paper).
# Set use_pr=false to train outcome-only GRPO.

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

ENGINE=${1:-vllm_osworld}

# ---------- User configuration ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_agpr}"
mkdir -p "${RAY_TMPDIR}"

N_NODES="${N_NODES:-1}"
N_GPUS="${N_GPUS:-4}"
N_GPUS_PER_NODE="${N_GPUS}"

MODEL_PATH="${MODEL_PATH:-/path/to/UI-TARS-1.5-7B}"
export ROLLOUTER_DATA_DIR="${ROLLOUTER_DATA_DIR:-dart_rollouter/results/osworld_rollout}"
export RUN_ID="${RUN_ID:-results/osworld_rollout}"
export ROLLOUT_SERVER_URL="${ROLLOUT_SERVER_URL:-http://127.0.0.1:15959}"

export DB_HOST="${DB_HOST:-127.0.0.1}"
export DB_USER="${DB_USER:-root}"
export DB_PASSWORD="${DB_PASSWORD:?Set DB_PASSWORD to the MySQL password}"
export DB_DATABASE="${DB_DATABASE:-dart}"
export DB_PORT="${DB_PORT:-3306}"
export DB_CHARSET="${DB_CHARSET:-utf8mb4}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DART-GUI-TRAIN_$(date +%Y%m%d_%H%M%S)_${RANDOM}}"

# ---------- Training hyperparameters ----------
adv_estimator=grpo
loss_mode=gspo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.1

clip_ratio_low=0.1
clip_ratio_high=0.28

max_prompt_length=32000
max_response_length=500
qwen3_enable_thinking=false
loss_agg_mode="seq-mean-token-mean"

train_bz_min=4
train_bz_max=8
train_prompt_bsz=4
train_prompt_mini_bsz=32

sp_size=4
use_dynamic_bsz=False
offload=False

limit_messages=35
splitter=stepwise
splitter_parallel=True
window_size=5
stride_size=5
max_steps=30

use_vllm_logp=True
use_sft_loss=False
use_token_ids_from_pt=True

# true: outcome advantage + group process advantage (paper). false: outcome GRPO.
use_pr=true
pr_adv_weight=0.5

actor_resume_ckpt_to_keep=1

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  TRAIN_LOGGER="['console','swanlab']"
else
  TRAIN_LOGGER="['console']"
fi

ray stop || true
mkdir -p logs
exec > >(tee "logs/${EXPERIMENT_NAME}_1.log") 2>&1

python3 -m verl.trainer.main_ppo_async \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=evaluation_examples/filtered_test_all.json \
    data.val_files=evaluation_examples/filtered_test_all.json \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=4 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    data.image_key=images \
    data.custom_cls.path=verl/utils/dataset/osworld_dataset_iter.py \
    data.custom_cls.name=OSWorldAsyncDataset \
    data.shuffle=false \
    +data.rotate_task_groups=true \
    +data.root_data_dir=$ROLLOUTER_DATA_DIR \
    +data.window_size=${window_size} \
    +data.stride_size=${stride_size} \
    +data.max_steps=${max_steps} \
    +data.num_workers=0 \
    +data.run_id=$RUN_ID \
    +data.steps_per_epoch=200 \
    +data.train_batch_size_min=${train_bz_min} \
    +data.train_batch_size_max=${train_bz_max} \
    +data.top_mvs_n=2 \
    +data.use_pr=${use_pr} \
    +data.pr_adv_weight=${pr_adv_weight} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    reward_model.reward_manager=osworld \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=2.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    "actor_rollout_ref.actor.checkpoint.save_contents=['model', 'optimizer', 'extra', 'hf_model']" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    +actor_rollout_ref.actor.use_vllm_logp=${use_vllm_logp} \
    +actor_rollout_ref.actor.use_sft_loss=${use_sft_loss} \
    +actor_rollout_ref.actor.use_token_ids_from_pt=${use_token_ids_from_pt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    trainer.logger=${TRAIN_LOGGER} \
    trainer.project_name='gui-process-reward' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=5 \
    +trainer.max_actor_resume_ckpt_to_keep=${actor_resume_ckpt_to_keep} \
    +trainer.run_id=$RUN_ID \
    +trainer.splitter=${splitter} \
    +trainer.limit_messages=${limit_messages} \
    +trainer.splitter_parallel=${splitter_parallel} \
    +trainer.qwen3_enable_thinking=${qwen3_enable_thinking} \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.top_k=200 \
    +actor_rollout_ref.rollout.max_steps=15 \
    +actor_rollout_ref.rollout.limit_images=5 \
    +actor_rollout_ref.rollout.server_url=$ROLLOUT_SERVER_URL \
    +actor_rollout_ref.actor.offline=false
