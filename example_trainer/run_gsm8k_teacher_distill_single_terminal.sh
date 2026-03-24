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
#   TRAINING_STEPS=10
#   DISTILL_COEF=0.2
#   DISTILL_TEMPERATURE=1.0
#   DIVERGENCE=forward_kl
#   JSD_BETA=0.1
#   TEACHER_TOP_K=4
#   REUSE_SERVERS=1              # Reuse healthy API + student + teacher servers
#   REUSE_API=1                  # Override reuse per service
#   REUSE_STUDENT=1
#   REUSE_TEACHER=1
#   STUDENT_SERVER_TYPE=vllm
#   TEACHER_SERVER_TYPE=vllm
#   SMOKE_TEST_GENERATE=1
#   DRY_RUN=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$PWD"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

STUDENT_GPUS="${STUDENT_GPUS:-0}"
TEACHER_GPUS="${TEACHER_GPUS:-4,5,6,7}"
TRAINER_GPUS="${TRAINER_GPUS:-$STUDENT_GPUS}"

STUDENT_TP="${STUDENT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-4}"
STUDENT_SERVER_TYPE="${STUDENT_SERVER_TYPE:-vllm}"
TEACHER_SERVER_TYPE="${TEACHER_SERVER_TYPE:-vllm}"

API_PORT="${API_PORT:-8002}"
STUDENT_PORT="${STUDENT_PORT:-9001}"
TEACHER_PORT="${TEACHER_PORT:-9003}"

TRAINING_STEPS="${TRAINING_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-12288}"
# Trainer seq_len must be larger than ENV_MAX_TOKEN_LENGTH to accommodate
# chat template overhead (~400-800 tokens for Qwen3 thinking format).
TRAINER_SEQ_LEN="${TRAINER_SEQ_LEN:-9216}"
ENV_MAX_TOKEN_LENGTH="${ENV_MAX_TOKEN_LENGTH:-8192}"
DISTILL_COEF="${DISTILL_COEF:-0.2}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
DIVERGENCE="${DIVERGENCE:-forward_kl}"
JSD_BETA="${JSD_BETA:-0.1}"
TEACHER_TOP_K="${TEACHER_TOP_K:-4}"

WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-teacher-distill}"
WANDB_GROUP="${WANDB_GROUP:-}"

STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.60}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
SAVE_DIR="${SAVE_DIR:-${LAUNCH_DIR}/saves/gsm8k_teacher_distill}"
LOG_DIR="${LOG_DIR:-${LAUNCH_DIR}/logs/gsm8k_teacher_distill}"
BRIDGE_DIR="${BRIDGE_DIR:-${LOG_DIR}/bridge}"
DRY_RUN="${DRY_RUN:-0}"
REUSE_SERVERS="${REUSE_SERVERS:-0}"
REUSE_API="${REUSE_API:-$REUSE_SERVERS}"
REUSE_STUDENT="${REUSE_STUDENT:-$REUSE_SERVERS}"
REUSE_TEACHER="${REUSE_TEACHER:-$REUSE_SERVERS}"
SMOKE_TEST_GENERATE="${SMOKE_TEST_GENERATE:-1}"

ENV_GROUP_SIZE="${ENV_GROUP_SIZE:-2}"
ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-1}"
ENV_TOTAL_STEPS="${ENV_TOTAL_STEPS:-20}"
ENV_STEPS_PER_EVAL="${ENV_STEPS_PER_EVAL:-10}"
ENV_MAX_WORKERS_PER_NODE="${ENV_MAX_WORKERS_PER_NODE:-1}"
ENV_WORKER_TIMEOUT="${ENV_WORKER_TIMEOUT:-3600}"

RUN_PIDS=()
RUN_PORTS=()
PID_DIR=""

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

pid_file_for() {
  local name="$1"
  printf '%s/%s.pid' "$PID_DIR" "$name"
}

write_pid_file() {
  local name="$1"
  local pid="$2"
  printf '%s\n' "$pid" >"$(pid_file_for "$name")"
}

remove_pid_file() {
  local name="$1"
  rm -f "$(pid_file_for "$name")"
}

pid_is_running() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

is_http_ready() {
  local url="$1"
  curl -fsS "$url" >/dev/null 2>&1
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

smoke_test_generate() {
  local port="$1"
  local name="$2"
  local timeout="${3:-120}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] skip /generate smoke test for ${name}"
    return 0
  fi

  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "http://localhost:${port}/generate" \
      -H 'Content-Type: application/json' \
      -d '{
        "prompt": "Reply with OK.",
        "max_tokens": 4,
        "temperature": 0.0,
        "n": 1,
        "logprobs": 0
      }' >/dev/null 2>&1; then
      log "Ready: ${name} /generate smoke test"
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      log "Timeout waiting for ${name} /generate smoke test"
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
  write_pid_file "$name" "$pid"
  log "${name} PID=${pid}"
}

stop_named_process() {
  local name="$1"
  local pid_file
  pid_file="$(pid_file_for "$name")"
  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi

  local pid
  pid="$(<"$pid_file")"
  if pid_is_running "$pid"; then
    log "Stopping ${name} PID=${pid}"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if pid_is_running "$pid"; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  remove_pid_file "$name"
}

stop_processes_matching() {
  local name="$1"
  local pattern="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] skip process cleanup for ${name} pattern=${pattern}"
    return 0
  fi

  local pids
  pids="$(pgrep -f -- "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if [[ "$pid" == "$$" ]]; then
      continue
    fi
    log "Stopping stale ${name} PID=${pid}"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if pid_is_running "$pid"; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done <<<"$pids"
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

PID_DIR="${LOG_DIR}/pids"
mkdir -p "$LOG_DIR" "$SAVE_DIR" "$BRIDGE_DIR" "$PID_DIR"

log "Config:"
log "  student=${STUDENT_MODEL}"
log "  teacher=${TEACHER_MODEL}"
log "  gpus student=${STUDENT_GPUS}, teacher=${TEACHER_GPUS}, trainer=${TRAINER_GPUS}"
log "  ports api=${API_PORT}, student=${STUDENT_PORT}, teacher=${TEACHER_PORT}"
log "  server types student=${STUDENT_SERVER_TYPE}, teacher=${TEACHER_SERVER_TYPE}"
log "  logs=${LOG_DIR}"
log "  saves=${SAVE_DIR}"
log "  bridge=${BRIDGE_DIR}"
log "  reuse api=${REUSE_API}, student=${REUSE_STUDENT}, teacher=${REUSE_TEACHER}"
log "  env max_token_length=${ENV_MAX_TOKEN_LENGTH}, env workers=${ENV_MAX_WORKERS_PER_NODE}, env worker_timeout=${ENV_WORKER_TIMEOUT}"
log "  wandb project=${WANDB_PROJECT}${WANDB_GROUP:+, group=${WANDB_GROUP}}"
log "  gkd divergence=${DIVERGENCE}${DIVERGENCE:+, jsd_beta=${JSD_BETA}}"

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

if [[ "$STUDENT_SERVER_TYPE" != "vllm" ]]; then
  log "ERROR: STUDENT_SERVER_TYPE must be vllm for GSM8K teacher-distill runner."
  exit 2
fi

if [[ "$TEACHER_SERVER_TYPE" != "vllm" ]]; then
  log "ERROR: TEACHER_SERVER_TYPE must be vllm for teacher logprob fetching."
  exit 2
fi

# 1) Atropos API
if [[ "$REUSE_API" == "1" && "$DRY_RUN" == "0" ]] && is_http_ready "http://localhost:${API_PORT}/info"; then
  log "Reusing existing run-api on :${API_PORT}"
else
  stop_named_process "run_api"
  RUN_PORTS+=("$API_PORT")
  kill_port "$API_PORT"
  start_process "run_api" "${LOG_DIR}/run_api.log" \
    run-api --port "$API_PORT"
  if [[ "$DRY_RUN" == "0" ]]; then
    wait_for_http "http://localhost:${API_PORT}/info" 180 "run-api"
  fi
fi

# 2) Student vLLM server
if [[ "$REUSE_STUDENT" == "1" && "$DRY_RUN" == "0" ]] && is_http_ready "http://localhost:${STUDENT_PORT}/health"; then
  log "Reusing existing student vLLM on :${STUDENT_PORT}"
else
  stop_named_process "student_vllm"
  RUN_PORTS+=("$STUDENT_PORT")
  kill_port "$STUDENT_PORT"
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
    if [[ "$SMOKE_TEST_GENERATE" == "1" ]]; then
      smoke_test_generate "$STUDENT_PORT" "student vLLM" 240
    fi
  fi
fi

# 3) Teacher vLLM server
if [[ "$REUSE_TEACHER" == "1" && "$DRY_RUN" == "0" ]] && is_http_ready "http://localhost:${TEACHER_PORT}/health"; then
  log "Reusing existing teacher vLLM on :${TEACHER_PORT}"
else
  stop_named_process "teacher_vllm"
  RUN_PORTS+=("$TEACHER_PORT")
  kill_port "$TEACHER_PORT"
  start_process "teacher_vllm" "${LOG_DIR}/teacher_vllm.log" \
    env CUDA_VISIBLE_DEVICES="$TEACHER_GPUS" \
    "$PYTHON_BIN" -m example_trainer.vllm_api_server \
      --model "$TEACHER_MODEL" \
      --port "$TEACHER_PORT" \
      --tensor-parallel-size "$TEACHER_TP" \
      --gpu-memory-utilization "$TEACHER_GPU_MEMORY_UTILIZATION" \
      --max-model-len "$TEACHER_MAX_MODEL_LEN" \
      --dtype "$DTYPE"
  if [[ "$DRY_RUN" == "0" ]]; then
    wait_for_http "http://localhost:${TEACHER_PORT}/health" 1800 "teacher vLLM"
    if [[ "$SMOKE_TEST_GENERATE" == "1" ]]; then
      smoke_test_generate "$TEACHER_PORT" "teacher vLLM" 300
    fi
  fi
fi

# 4) Teacher-distill GSM8K env
stop_named_process "gsm8k_teacher_env"
stop_processes_matching "gsm8k_teacher_env" "environments/gsm8k_server_teacher_distill.py serve"
start_process "gsm8k_teacher_env" "${LOG_DIR}/env.log" \
  "$PYTHON_BIN" environments/gsm8k_server_teacher_distill.py serve \
    --env.tokenizer_name "$STUDENT_MODEL" \
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
    --teacher.base_url "http://localhost:${TEACHER_PORT}/v1" \
    --teacher.model_name "$TEACHER_MODEL" \
    --teacher.tokenizer_name "$STUDENT_MODEL" \
    --teacher.server_type "$TEACHER_SERVER_TYPE" \
    --env.teacher_top_k "$TEACHER_TOP_K" \
    --env.ensure_scores_are_not_same false \
    --openai.api_key "dummy" \
    --openai.base_url "http://localhost:${STUDENT_PORT}/v1" \
    --openai.model_name "$STUDENT_MODEL" \
    --openai.tokenizer_name "$STUDENT_MODEL" \
    --openai.server_type "$STUDENT_SERVER_TYPE"

log "All services launched."
log "Run logs:"
log "  ${LOG_DIR}/run_api.log"
log "  ${LOG_DIR}/student_vllm.log"
log "  ${LOG_DIR}/teacher_vllm.log"
log "  ${LOG_DIR}/env.log"

# 5) Trainer (background)
TRAINER_CMD=(
  env
  CUDA_VISIBLE_DEVICES="$TRAINER_GPUS"
  PYTHONUNBUFFERED=1
  "$PYTHON_BIN"
  -u
  -m
  example_trainer.gkd
  --model-name "$STUDENT_MODEL"
  --weight-bridge-mode shared_vllm
  --device cuda:0
  --save-path "$SAVE_DIR"
  --atropos-url "http://localhost:${API_PORT}"
  --vllm-port "$STUDENT_PORT"
  --vllm-config-path "${BRIDGE_DIR}/vllm_bridge_config.json"
  --training-steps "$TRAINING_STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --warmup-steps "$WARMUP_STEPS"
  --lr "$LR"
  --seq-len "$TRAINER_SEQ_LEN"
  --distill-coef "$DISTILL_COEF"
  --distill-temperature "$DISTILL_TEMPERATURE"
  --divergence "$DIVERGENCE"
  --jsd-beta "$JSD_BETA"
  --use-wandb
  --wandb-project "$WANDB_PROJECT"
)
if [[ -n "$WANDB_GROUP" ]]; then
  TRAINER_CMD+=(--wandb-group "$WANDB_GROUP")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] trainer command:"
  printf '  '
  printf '%q ' "${TRAINER_CMD[@]}"
  printf '\n'
  exit 0
fi

stop_named_process "trainer"
stop_processes_matching "trainer" "example_trainer.gkd"
start_process "trainer" "${LOG_DIR}/trainer.log" "${TRAINER_CMD[@]}"

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
log "  curl -s http://localhost:${STUDENT_PORT}/debug/request_stats | jq ."
log "  curl -s \"http://localhost:${STUDENT_PORT}/debug/request_stats?stuck_threshold_s=120\" | jq ."
log "  curl -s http://localhost:${STUDENT_PORT}/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Reply with OK.\",\"max_tokens\":4,\"temperature\":0.0,\"n\":1,\"logprobs\":0}' | jq ."
log "  curl -s http://localhost:${TEACHER_PORT}/debug/request_stats | jq ."
log "  curl -s \"http://localhost:${TEACHER_PORT}/debug/request_stats?stuck_threshold_s=120\" | jq ."
log "  curl -s http://localhost:${TEACHER_PORT}/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Reply with OK.\",\"max_tokens\":4,\"temperature\":0.0,\"n\":1,\"logprobs\":0}' | jq ."
log ""
log "Reuse tip:"
log "  REUSE_SERVERS=1 ./example_trainer/run_gsm8k_teacher_distill_single_terminal.sh"
log "  This keeps healthy API/student/teacher servers and only restarts env + trainer."
log ""
log "To stop processes started by this invocation:"
log "  kill ${RUN_PIDS[*]:-} 2>/dev/null; sleep 1; kill -9 ${RUN_PIDS[*]:-} 2>/dev/null"
trap - EXIT INT TERM
