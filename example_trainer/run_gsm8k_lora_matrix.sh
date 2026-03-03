#!/usr/bin/env bash
set -euo pipefail

# Runs three GSM8K test trainings with separate infra/ports:
#   1) shared_vllm
#   2) lora_only
#   3) lora_restart
#
# Usage:
#   chmod +x example_trainer/run_gsm8k_lora_matrix.sh
#   ./example_trainer/run_gsm8k_lora_matrix.sh
#
# Optional environment overrides:
#   MODEL_NAME="NousResearch/Hermes-3-Llama-3.1-8B"
#   TRAINING_STEPS=30
#   WARMUP_STEPS=5
#   MATRIX_TARGETED=1          # auto-enable layer targeting defaults for smoke tests
#   SHARED_LAYER_INDICES="0-3" # overrides MATRIX_TARGETED default
#   LORA_LAYER_INDICES="0-3"   # overrides MATRIX_TARGETED default
#   WANDB_PROJECT="gsm8k-grpo-smoke"
#   WANDB_GROUP="gsm8k-$(date +%Y%m%d-%H%M%S)"
#   START_API_PORT=8002
#   START_VLLM_PORT=9001
#   PYTHON_BIN=python3
#   OUTPUT_BASE_DIR="$PWD"   # logs/saves base (defaults to launch directory)
#   SHARED_GPU_MEMORY_UTILIZATION=0.60  # shared_vllm only (H100-friendly default)
#   SHARED_GPU=0
#   LORA_ONLY_TRAINER_GPU=1
#   LORA_ONLY_VLLM_GPU=2
#   LORA_RESTART_TRAINER_GPU=3
#   LORA_RESTART_VLLM_GPU=4
#   DRY_RUN=1                # print commands only, do not execute
#   PARALLEL=1               # run all three modes concurrently
#   MODE=all                 # one of: all, shared_vllm, lora_only, lora_restart

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
LAUNCH_DIR="$PWD"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Hermes-3-Llama-3.1-8B}"
TRAINING_STEPS="${TRAINING_STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
CLIP_EPS="${CLIP_EPS:-0.2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
SHARED_GPU_MEMORY_UTILIZATION="${SHARED_GPU_MEMORY_UTILIZATION:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj v_proj}"
MATRIX_TARGETED="${MATRIX_TARGETED:-1}"
SHARED_LAYER_INDICES="${SHARED_LAYER_INDICES:-}"
LORA_LAYER_INDICES="${LORA_LAYER_INDICES:-}"
if [[ "$MATRIX_TARGETED" == "1" ]]; then
  SHARED_LAYER_INDICES="${SHARED_LAYER_INDICES:-0-3}"
  LORA_LAYER_INDICES="${LORA_LAYER_INDICES:-0-3}"
fi
WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-grpo-smoke}"
WANDB_GROUP="${WANDB_GROUP:-gsm8k-$(date +%Y%m%d-%H%M%S)}"
START_API_PORT="${START_API_PORT:-8002}"
START_VLLM_PORT="${START_VLLM_PORT:-9001}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$LAUNCH_DIR}"

# GPU pinning (one process per GPU preference)
SHARED_GPU="${SHARED_GPU:-0}"
LORA_ONLY_TRAINER_GPU="${LORA_ONLY_TRAINER_GPU:-1}"
LORA_ONLY_VLLM_GPU="${LORA_ONLY_VLLM_GPU:-2}"
LORA_RESTART_TRAINER_GPU="${LORA_RESTART_TRAINER_GPU:-3}"
LORA_RESTART_VLLM_GPU="${LORA_RESTART_VLLM_GPU:-4}"
DRY_RUN="${DRY_RUN:-0}"
ENV_TOTAL_STEPS="${ENV_TOTAL_STEPS:-200}"
ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-16}"
ENV_MAX_WORKERS_PER_NODE="${ENV_MAX_WORKERS_PER_NODE:-8}"
ENV_STEPS_PER_EVAL="${ENV_STEPS_PER_EVAL:-50}"
PARALLEL="${PARALLEL:-0}"
MODE="${MODE:-all}"

SHARED_API_PORT="$START_API_PORT"
SHARED_VLLM_PORT="$START_VLLM_PORT"
LORA_ONLY_API_PORT="$((START_API_PORT + 1))"
LORA_ONLY_VLLM_PORT="$((START_VLLM_PORT + 1))"
LORA_RESTART_API_PORT="$((START_API_PORT + 2))"
LORA_RESTART_VLLM_PORT="$((START_VLLM_PORT + 2))"

run_pids=()
run_ports=()

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

kill_port() {
  local port="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] skip port cleanup for :${port}"
    return 0
  fi
  if lsof -i ":${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -ti ":${port}" | xargs -r kill -9 || true
  fi
}

wait_for_http() {
  local url="$1"
  local timeout="${2:-180}"
  local name="${3:-endpoint}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "Ready: ${name} (${url})"
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      log "Timeout waiting for ${name}: ${url}"
      return 1
    fi
    sleep 2
  done
}

start_process() {
  local name="$1"
  local logfile="$2"
  shift 2
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] start ${name} (log: ${logfile})"
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  log "Starting ${name} (log: ${logfile})"
  "$@" >"$logfile" 2>&1 &
  local pid=$!
  run_pids+=("$pid")
  log "${name} PID=${pid}"
}

cleanup_run() {
  log "Cleaning up run processes..."
  if (( ${#run_pids[@]} > 0 )); then
    for pid in "${run_pids[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
    sleep 1
    for pid in "${run_pids[@]}"; do
      kill -9 "$pid" >/dev/null 2>&1 || true
    done
  fi
  if (( ${#run_ports[@]} > 0 )); then
    for port in "${run_ports[@]}"; do
      kill_port "$port"
    done
  fi
  run_pids=()
  run_ports=()
}

add_shared_layer_flag() {
  if [[ -n "$SHARED_LAYER_INDICES" ]]; then
    echo "--train-layer-indices" "$SHARED_LAYER_INDICES"
  fi
}

add_lora_layer_flag() {
  if [[ -n "$LORA_LAYER_INDICES" ]]; then
    echo "--lora-layer-indices" "$LORA_LAYER_INDICES"
  fi
}

common_trainer_flags() {
  echo \
    --model-name "$MODEL_NAME" \
    --training-steps "$TRAINING_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --warmup-steps "$WARMUP_STEPS" \
    --lr "$LR" \
    --clip-eps "$CLIP_EPS" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP"
}

start_gsm8k_env() {
  local api_port="$1"
  local vllm_port="$2"
  local env_wandb_name="$3"
  local logfile="$4"
  start_process "gsm8k_env" "$logfile" \
    "$PYTHON_BIN" environments/gsm8k_server.py serve \
      --env.group_size 4 \
      --env.batch_size "$ENV_BATCH_SIZE" \
      --env.total_steps "$ENV_TOTAL_STEPS" \
      --env.steps_per_eval "$ENV_STEPS_PER_EVAL" \
      --env.max_num_workers_per_node "$ENV_MAX_WORKERS_PER_NODE" \
      --env.max_token_length "$MAX_MODEL_LEN" \
      --env.rollout_server_url "http://localhost:${api_port}" \
      --env.use_wandb true \
      --env.wandb_name "$env_wandb_name" \
      --openai.api_key "dummy" \
      --openai.base_url "http://localhost:${vllm_port}/v1" \
      --openai.model_name "$MODEL_NAME" \
      --openai.server_type vllm
}

start_gsm8k_env_shared() {
  local vllm_port="$1"
  local logfile="$2"
  local api_port="$SHARED_API_PORT"
  start_gsm8k_env "$api_port" "$vllm_port" "gsm8k-shared-vllm-env" "$logfile"
}

run_shared_vllm() {
  log "========== RUN: shared_vllm =========="
  local api_port="$SHARED_API_PORT"
  local vllm_port="$SHARED_VLLM_PORT"
  local mode_dir="${OUTPUT_BASE_DIR}/logs/gsm8k_shared_vllm"
  local save_dir="${OUTPUT_BASE_DIR}/saves/gsm8k_shared_vllm"
  local bridge_dir="${mode_dir}/bridge"
  mkdir -p "$mode_dir"
  mkdir -p "$save_dir"
  mkdir -p "$bridge_dir"

  run_ports+=("$api_port" "$vllm_port")
  kill_port "$api_port"
  kill_port "$vllm_port"

  start_process "run_api" "$mode_dir/run_api.log" run-api --port "$api_port"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] wait for http://localhost:${api_port}/info"
  else
    wait_for_http "http://localhost:${api_port}/info" 60 "run-api"
  fi

  start_process "vllm_shared" "$mode_dir/vllm.log" \
    env CUDA_VISIBLE_DEVICES="$SHARED_GPU" VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR="$bridge_dir" \
    "$PYTHON_BIN" -m example_trainer.vllm_api_server \
      --model "$MODEL_NAME" \
      --port "$vllm_port" \
      --gpu-memory-utilization "$SHARED_GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --dtype "$DTYPE" \
      --enforce-eager
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] wait for http://localhost:${vllm_port}/health"
  else
    wait_for_http "http://localhost:${vllm_port}/health" 300 "shared vLLM"
  fi

  start_gsm8k_env_shared "$vllm_port" "$mode_dir/env.log"

  log "Starting trainer: shared_vllm"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] trainer command (shared_vllm):"
    printf '  '
    printf '%q ' env CUDA_VISIBLE_DEVICES="$SHARED_GPU" "$PYTHON_BIN" -m example_trainer.grpo \
      $(common_trainer_flags) \
      --weight-bridge-mode shared_vllm \
      --device cuda:0 \
      --save-path "$save_dir" \
      --vllm-port "$vllm_port" \
      --vllm-config-path "${bridge_dir}/vllm_bridge_config.json" \
      --atropos-url "http://localhost:${api_port}" \
      $(add_shared_layer_flag)
    printf '\n'
    log "[DRY RUN] trainer log path: $mode_dir/trainer.log"
  else
    env CUDA_VISIBLE_DEVICES="$SHARED_GPU" "$PYTHON_BIN" -m example_trainer.grpo \
      $(common_trainer_flags) \
      --weight-bridge-mode shared_vllm \
      --device cuda:0 \
      --save-path "$save_dir" \
      --vllm-port "$vllm_port" \
      --vllm-config-path "${bridge_dir}/vllm_bridge_config.json" \
      --atropos-url "http://localhost:${api_port}" \
      $(add_shared_layer_flag) | tee "$mode_dir/trainer.log"
  fi

  cleanup_run
}

run_lora_only() {
  log "========== RUN: lora_only =========="
  local api_port="$LORA_ONLY_API_PORT"
  local vllm_port="$LORA_ONLY_VLLM_PORT"
  local mode_dir="${OUTPUT_BASE_DIR}/logs/gsm8k_lora_only"
  local save_dir="${OUTPUT_BASE_DIR}/saves/gsm8k_lora_only"
  mkdir -p "$mode_dir"
  mkdir -p "$save_dir"

  run_ports+=("$api_port" "$vllm_port")
  kill_port "$api_port"
  kill_port "$vllm_port"

  start_process "run_api" "$mode_dir/run_api.log" run-api --port "$api_port"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] wait for http://localhost:${api_port}/info"
  else
    wait_for_http "http://localhost:${api_port}/info" 60 "run-api"
  fi

  start_process "vllm_lora_only" "$mode_dir/vllm.log" \
    env CUDA_VISIBLE_DEVICES="$LORA_ONLY_VLLM_GPU" \
    "$PYTHON_BIN" -m example_trainer.vllm_api_server \
      --model "$MODEL_NAME" \
      --port "$vllm_port" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --dtype "$DTYPE" \
      --enable-lora \
      --enforce-eager
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] wait for http://localhost:${vllm_port}/health"
  else
    wait_for_http "http://localhost:${vllm_port}/health" 300 "lora_only vLLM"
  fi

  start_gsm8k_env "$api_port" "$vllm_port" "gsm8k-lora-only-env" "$mode_dir/env.log"

  log "Starting trainer: lora_only"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] trainer command (lora_only):"
    printf '  '
    printf '%q ' env CUDA_VISIBLE_DEVICES="$LORA_ONLY_TRAINER_GPU" "$PYTHON_BIN" -m example_trainer.grpo \
      $(common_trainer_flags) \
      --weight-bridge-mode lora_only \
      --device cuda:0 \
      --save-path "$save_dir" \
      --vllm-port "$vllm_port" \
      --atropos-url "http://localhost:${api_port}" \
      --lora-r "$LORA_R" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-target-modules $LORA_TARGET_MODULES \
      $(add_lora_layer_flag)
    printf '\n'
    log "[DRY RUN] trainer log path: $mode_dir/trainer.log"
  else
    env CUDA_VISIBLE_DEVICES="$LORA_ONLY_TRAINER_GPU" "$PYTHON_BIN" -m example_trainer.grpo \
      $(common_trainer_flags) \
      --weight-bridge-mode lora_only \
      --device cuda:0 \
      --save-path "$save_dir" \
      --vllm-port "$vllm_port" \
      --atropos-url "http://localhost:${api_port}" \
      --lora-r "$LORA_R" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-target-modules $LORA_TARGET_MODULES \
      $(add_lora_layer_flag) | tee "$mode_dir/trainer.log"
  fi

  cleanup_run
}

run_lora_restart() {
  log "========== RUN: lora_restart =========="
  local api_port="$LORA_RESTART_API_PORT"
  local vllm_port="$LORA_RESTART_VLLM_PORT"
  local mode_dir="${OUTPUT_BASE_DIR}/logs/gsm8k_lora_restart"
  local save_dir="${OUTPUT_BASE_DIR}/saves/gsm8k_lora_restart"
  mkdir -p "$mode_dir"
  mkdir -p "$save_dir"

  run_ports+=("$api_port" "$vllm_port")
  kill_port "$api_port"
  kill_port "$vllm_port"

  start_process "run_api" "$mode_dir/run_api.log" run-api --port "$api_port"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] wait for http://localhost:${api_port}/info"
  else
    wait_for_http "http://localhost:${api_port}/info" 60 "run-api"
  fi

  log "Starting trainer: lora_restart (it launches its own vLLM)"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] trainer command (lora_restart):"
    printf '  '
    printf '%q ' env CUDA_VISIBLE_DEVICES="$LORA_RESTART_TRAINER_GPU" "$PYTHON_BIN" -m example_trainer.grpo \
      $(common_trainer_flags) \
      --weight-bridge-mode lora_restart \
      --device cuda:0 \
      --save-path "$save_dir" \
      --vllm-port "$vllm_port" \
      --vllm-gpu "$LORA_RESTART_VLLM_GPU" \
      --vllm-restart-interval 3 \
      --atropos-url "http://localhost:${api_port}" \
      --lora-r "$LORA_R" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-target-modules $LORA_TARGET_MODULES \
      $(add_lora_layer_flag)
    printf '\n'
    log "[DRY RUN] then wait for http://localhost:${vllm_port}/health"
    log "[DRY RUN] then start GSM8K env pointed at http://localhost:${vllm_port}/v1 and rollout server http://localhost:${api_port}"
    log "[DRY RUN] trainer log path: $mode_dir/trainer.log"
  else
    env CUDA_VISIBLE_DEVICES="$LORA_RESTART_TRAINER_GPU" "$PYTHON_BIN" -m example_trainer.grpo \
      $(common_trainer_flags) \
      --weight-bridge-mode lora_restart \
      --device cuda:0 \
      --save-path "$save_dir" \
      --vllm-port "$vllm_port" \
      --vllm-gpu "$LORA_RESTART_VLLM_GPU" \
      --vllm-restart-interval 3 \
      --atropos-url "http://localhost:${api_port}" \
      --lora-r "$LORA_R" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-target-modules $LORA_TARGET_MODULES \
      $(add_lora_layer_flag) >"$mode_dir/trainer.log" 2>&1 &
    trainer_pid=$!
    run_pids+=("$trainer_pid")

    wait_for_http "http://localhost:${vllm_port}/health" 420 "lora_restart vLLM"
    start_gsm8k_env "$api_port" "$vllm_port" "gsm8k-lora-restart-env" "$mode_dir/env.log"

    wait "$trainer_pid"
    cat "$mode_dir/trainer.log"
  fi

  cleanup_run
}

trap cleanup_run EXIT INT TERM

log "Model: $MODEL_NAME"
log "W&B project/group: $WANDB_PROJECT / $WANDB_GROUP"
log "Dry run mode: $DRY_RUN"
log "Output base directory (logs + saves): $OUTPUT_BASE_DIR"
log "Warmup steps: $WARMUP_STEPS"
log "Targeted-layer matrix profile: $MATRIX_TARGETED"
log "vLLM memory utilization: shared=${SHARED_GPU_MEMORY_UTILIZATION}, lora=${GPU_MEMORY_UTILIZATION}"
log "Port plan:"
log "  shared_vllm:   run-api=${SHARED_API_PORT}, vllm=${SHARED_VLLM_PORT}"
log "  lora_only:     run-api=${LORA_ONLY_API_PORT}, vllm=${LORA_ONLY_VLLM_PORT}"
log "  lora_restart:  run-api=${LORA_RESTART_API_PORT}, vllm=${LORA_RESTART_VLLM_PORT}"
log "GPU plan:"
log "  shared_vllm:   trainer+vllm on GPU ${SHARED_GPU} (required for shared weights)"
log "  lora_only:     trainer GPU ${LORA_ONLY_TRAINER_GPU}, vllm GPU ${LORA_ONLY_VLLM_GPU}"
log "  lora_restart:  trainer GPU ${LORA_RESTART_TRAINER_GPU}, vllm GPU ${LORA_RESTART_VLLM_GPU}"
if [[ -n "$SHARED_LAYER_INDICES" ]]; then
  log "Shared-model train layer indices: $SHARED_LAYER_INDICES"
else
  log "Shared-model train layer indices: all layers"
fi
if [[ -n "$LORA_LAYER_INDICES" ]]; then
  log "LoRA layer indices: $LORA_LAYER_INDICES"
else
  log "LoRA layer indices: all matching layers"
fi
log "Mode selector: $MODE"
log "Parallel mode: $PARALLEL"

if [[ "$MODE" != "all" ]]; then
  case "$MODE" in
    shared_vllm) run_shared_vllm ;;
    lora_only) run_lora_only ;;
    lora_restart) run_lora_restart ;;
    *)
      log "Invalid MODE='$MODE' (expected: all|shared_vllm|lora_only|lora_restart)"
      exit 2
      ;;
  esac
  log "Mode '$MODE' completed."
  exit 0
fi

if [[ "$PARALLEL" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] parallel launcher commands:"
    for m in shared_vllm lora_only lora_restart; do
      local_log="${OUTPUT_BASE_DIR}/logs/gsm8k_${m}/orchestrator.log"
      printf '  '
      printf '%q ' env MODE="$m" PARALLEL=0 "$SCRIPT_PATH"
      printf '> %q 2>&1 &\n' "$local_log"
    done
    log "[DRY RUN] parent waits for all child mode runners."
  else
    log "Launching all modes in parallel..."
    parallel_pids=()
    parallel_modes=(shared_vllm lora_only lora_restart)
    for m in "${parallel_modes[@]}"; do
      mode_log_dir="${OUTPUT_BASE_DIR}/logs/gsm8k_${m}"
      mkdir -p "$mode_log_dir"
      mode_orch_log="${mode_log_dir}/orchestrator.log"
      log "Starting mode runner: ${m} (log: ${mode_orch_log})"
      env MODE="$m" PARALLEL=0 "$SCRIPT_PATH" >"$mode_orch_log" 2>&1 &
      parallel_pids+=("$!")
    done

    fail_count=0
    for i in "${!parallel_pids[@]}"; do
      pid="${parallel_pids[$i]}"
      mode="${parallel_modes[$i]}"
      if wait "$pid"; then
        log "Mode '${mode}' finished successfully."
      else
        log "Mode '${mode}' failed. See ${OUTPUT_BASE_DIR}/logs/gsm8k_${mode}/orchestrator.log"
        fail_count=$((fail_count + 1))
      fi
    done
    if (( fail_count > 0 )); then
      log "Parallel run finished with ${fail_count} failed mode(s)."
      exit 1
    fi
  fi
else
  run_shared_vllm
  run_lora_only
  run_lora_restart
fi

log "All GSM8K mode runs completed."
