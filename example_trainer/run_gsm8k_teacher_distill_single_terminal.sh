#!/usr/bin/env bash
set -euo pipefail

# Single-terminal teacher-distillation runner.
# Starts everything in the background from ONE shell that has GPU access:
#   1) Atropos API
#   2) Student vLLM server
#   3) Teacher vLLM server
#   4) GSM8K teacher-distill environment
#   5) Example trainer (foreground)
#
# Usage:
#   chmod +x example_trainer/run_gsm8k_teacher_distill_single_terminal.sh
#   ./example_trainer/run_gsm8k_teacher_distill_single_terminal.sh
#
# Optional overrides:
#   STUDENT_MODEL="Qwen/Qwen3-4B-Instruct-2507-FP8"
#   TEACHER_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
#   STUDENT_GPUS="0"
#   TEACHER_GPUS="4,5,6,7"
#   TRAINER_GPUS="0"
#   STUDENT_TP=1
#   TEACHER_TP=4
#   API_PORT=8002
#   STUDENT_PORT=9001
#   TEACHER_PORT=9003
#   TRAINING_STEPS=100
#   DISTILL_COEF=0.2
#   DISTILL_TEMPERATURE=1.0
#   TEACHER_TOP_K=8
#   DRY_RUN=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$PWD"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

STUDENT_GPUS="${STUDENT_GPUS:-0}"
TEACHER_GPUS="${TEACHER_GPUS:-4,5,6,7}"
TRAINER_GPUS="${TRAINER_GPUS:-$STUDENT_GPUS}"

STUDENT_TP="${STUDENT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-4}"

API_PORT="${API_PORT:-8002}"
STUDENT_PORT="${STUDENT_PORT:-9001}"
TEACHER_PORT="${TEACHER_PORT:-9003}"

TRAINING_STEPS="${TRAINING_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ENV_MAX_TOKEN_LENGTH="${ENV_MAX_TOKEN_LENGTH:-8192}"
DISTILL_COEF="${DISTILL_COEF:-0.2}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
TEACHER_TOP_K="${TEACHER_TOP_K:-8}"

STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.95}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
SAVE_DIR="${SAVE_DIR:-${LAUNCH_DIR}/saves/gsm8k_teacher_distill}"
LOG_DIR="${LOG_DIR:-${LAUNCH_DIR}/logs/gsm8k_teacher_distill}"
BRIDGE_DIR="${BRIDGE_DIR:-${LOG_DIR}/bridge}"
DRY_RUN="${DRY_RUN:-0}"

ENV_GROUP_SIZE="${ENV_GROUP_SIZE:-4}"
ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-16}"
ENV_TOTAL_STEPS="${ENV_TOTAL_STEPS:-200}"
ENV_STEPS_PER_EVAL="${ENV_STEPS_PER_EVAL:-50}"
ENV_MAX_WORKERS_PER_NODE="${ENV_MAX_WORKERS_PER_NODE:-1}"
ENV_WORKER_TIMEOUT="${ENV_WORKER_TIMEOUT:-300}"

RUN_PIDS=()
RUN_PORTS=()

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
  local timeout="${2:-240}"
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
  RUN_PIDS+=("$pid")
  log "${name} PID=${pid}"
}

cleanup_all() {
  log "Cleaning up processes..."
  for pid in "${RUN_PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  sleep 1
  for pid in "${RUN_PIDS[@]:-}"; do
    kill -9 "$pid" >/dev/null 2>&1 || true
  done
  for port in "${RUN_PORTS[@]:-}"; do
    kill_port "$port"
  done
}

trap cleanup_all EXIT INT TERM

mkdir -p "$LOG_DIR" "$SAVE_DIR" "$BRIDGE_DIR"
RUN_PORTS+=("$API_PORT" "$STUDENT_PORT" "$TEACHER_PORT")
kill_port "$API_PORT"
kill_port "$STUDENT_PORT"
kill_port "$TEACHER_PORT"

log "Config:"
log "  student=${STUDENT_MODEL}"
log "  teacher=${TEACHER_MODEL}"
log "  gpus student=${STUDENT_GPUS}, teacher=${TEACHER_GPUS}, trainer=${TRAINER_GPUS}"
log "  ports api=${API_PORT}, student=${STUDENT_PORT}, teacher=${TEACHER_PORT}"
log "  logs=${LOG_DIR}"
log "  saves=${SAVE_DIR}"
log "  bridge=${BRIDGE_DIR}"
log "  env max_token_length=${ENV_MAX_TOKEN_LENGTH}, env workers=${ENV_MAX_WORKERS_PER_NODE}, env worker_timeout=${ENV_WORKER_TIMEOUT}"

# Shared-vLLM attach path currently expects the student server to expose
# unsharded weights. Keep the student on TP=1 and the trainer on the same GPU set.
if [[ "$STUDENT_TP" != "1" ]]; then
  log "ERROR: shared_vllm teacher-distill runner currently requires STUDENT_TP=1."
  log "       The current attach path does not support TP-sharded student bridge weights."
  exit 2
fi

if [[ "$TRAINER_GPUS" != "$STUDENT_GPUS" ]]; then
  log "ERROR: TRAINER_GPUS must match STUDENT_GPUS for shared_vllm mode."
  log "       Got student=${STUDENT_GPUS}, trainer=${TRAINER_GPUS}"
  exit 2
fi

# 1) Atropos API
start_process "run_api" "${LOG_DIR}/run_api.log" \
  run-api --port "$API_PORT"
if [[ "$DRY_RUN" == "0" ]]; then
  wait_for_http "http://localhost:${API_PORT}/info" 180 "run-api"
fi

# 2) Student vLLM server
start_process "student_vllm" "${LOG_DIR}/student_vllm.log" \
  env CUDA_VISIBLE_DEVICES="$STUDENT_GPUS" VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR="$BRIDGE_DIR" \
  "$PYTHON_BIN" -m example_trainer.vllm_api_server \
    --model "$STUDENT_MODEL" \
    --port "$STUDENT_PORT" \
    --tensor-parallel-size "$STUDENT_TP" \
    --gpu-memory-utilization "$STUDENT_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE"
if [[ "$DRY_RUN" == "0" ]]; then
  wait_for_http "http://localhost:${STUDENT_PORT}/health" 420 "student vLLM"
fi

# 3) Teacher vLLM server
start_process "teacher_vllm" "${LOG_DIR}/teacher_vllm.log" \
  env CUDA_VISIBLE_DEVICES="$TEACHER_GPUS" \
  "$PYTHON_BIN" -m example_trainer.vllm_api_server \
    --model "$TEACHER_MODEL" \
    --port "$TEACHER_PORT" \
    --tensor-parallel-size "$TEACHER_TP" \
    --gpu-memory-utilization "$TEACHER_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE"
if [[ "$DRY_RUN" == "0" ]]; then
  wait_for_http "http://localhost:${TEACHER_PORT}/health" 1800 "teacher vLLM"
fi

# 4) Teacher-distill GSM8K env
start_process "gsm8k_teacher_env" "${LOG_DIR}/env.log" \
  "$PYTHON_BIN" environments/gsm8k_server_teacher_distill.py serve \
    --env.group_size "$ENV_GROUP_SIZE" \
    --env.batch_size "$ENV_BATCH_SIZE" \
    --env.total_steps "$ENV_TOTAL_STEPS" \
    --env.steps_per_eval "$ENV_STEPS_PER_EVAL" \
    --env.max_num_workers_per_node "$ENV_MAX_WORKERS_PER_NODE" \
    --env.max_token_length "$ENV_MAX_TOKEN_LENGTH" \
    --env.worker_timeout "$ENV_WORKER_TIMEOUT" \
    --env.rollout_server_url "http://localhost:${API_PORT}" \
    --env.use_wandb true \
    --env.wandb_name "gsm8k-teacher-distill" \
    --env.teacher_enabled true \
    --env.teacher_base_url "http://localhost:${TEACHER_PORT}/v1" \
    --env.teacher_model_name "$TEACHER_MODEL" \
    --env.teacher_top_k "$TEACHER_TOP_K" \
    --openai.api_key "dummy" \
    --openai.base_url "http://localhost:${STUDENT_PORT}/v1" \
    --openai.model_name "$STUDENT_MODEL" \
    --openai.tokenizer_name "$STUDENT_MODEL" \
    --openai.server_type vllm

log "All services launched."
log "Run logs:"
log "  ${LOG_DIR}/run_api.log"
log "  ${LOG_DIR}/student_vllm.log"
log "  ${LOG_DIR}/teacher_vllm.log"
log "  ${LOG_DIR}/env.log"

# 5) Trainer (foreground, primary output)
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] trainer command:"
  printf '  '
  printf '%q ' env CUDA_VISIBLE_DEVICES="$TRAINER_GPUS" \
    "$PYTHON_BIN" -m example_trainer.grpo \
    --model-name "$STUDENT_MODEL" \
    --weight-bridge-mode shared_vllm \
    --device cuda:0 \
    --save-path "$SAVE_DIR" \
    --atropos-url "http://localhost:${API_PORT}" \
    --vllm-port "$STUDENT_PORT" \
    --vllm-config-path "${BRIDGE_DIR}/vllm_bridge_config.json" \
    --training-steps "$TRAINING_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --warmup-steps "$WARMUP_STEPS" \
    --lr "$LR" \
    --clip-eps "$CLIP_EPS" \
    --distill-enabled \
    --distill-coef "$DISTILL_COEF" \
    --distill-temperature "$DISTILL_TEMPERATURE"
  printf '\n'
  exit 0
fi

start_process "trainer" "${LOG_DIR}/trainer.log" \
  env CUDA_VISIBLE_DEVICES="$TRAINER_GPUS" \
  "$PYTHON_BIN" -m example_trainer.grpo \
    --model-name "$STUDENT_MODEL" \
    --weight-bridge-mode shared_vllm \
    --device cuda:0 \
    --save-path "$SAVE_DIR" \
    --atropos-url "http://localhost:${API_PORT}" \
    --vllm-port "$STUDENT_PORT" \
    --vllm-config-path "${BRIDGE_DIR}/vllm_bridge_config.json" \
    --training-steps "$TRAINING_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --warmup-steps "$WARMUP_STEPS" \
    --lr "$LR" \
    --clip-eps "$CLIP_EPS" \
    --distill-enabled \
    --distill-coef "$DISTILL_COEF" \
    --distill-temperature "$DISTILL_TEMPERATURE"

log "All processes running in background."
log ""
log "Monitor with:"
log "  tail -f ${LOG_DIR}/trainer.log"
log "  tail -f ${LOG_DIR}/env.log"
log "  tail -f ${LOG_DIR}/student_vllm.log"
log "  tail -f ${LOG_DIR}/teacher_vllm.log"
log ""
log "Test endpoints:"
log "  curl -s http://localhost:${STUDENT_PORT}/health"
log "  curl -s http://localhost:${TEACHER_PORT}/health"
log "  curl -s http://localhost:${STUDENT_PORT}/bridge/is_paused | jq ."
log ""
log "To stop all processes:"
log "  kill ${RUN_PIDS[*]:-} 2>/dev/null; sleep 1; kill -9 ${RUN_PIDS[*]:-} 2>/dev/null"
trap - EXIT INT TERM
