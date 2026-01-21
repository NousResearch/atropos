# HYDRA + DRQ + Atropos Integration

**Purpose:** Bridge the gap between HYDRA's vision (security specialists) and the execution platform (Atropos RL framework) using DRQ's core insight (adversarial co-evolution beats static benchmarks).

**Status:** This document fixes the major disconnect: HYDRA plans assume "some offline trainer" but don't use Atropos (the RL framework in this repo) or DRQ (the original inspiration).

---

## The Gap

| What HYDRA Plans Say | What's Actually Here | What's Missing |
|---------------------|---------------------|----------------|
| "Train via DPO/GRPO refresh" | Atropos has `BaseEnv` + rollout server + trainer integration | Plans don't reference Atropos |
| "Verifier harness is the moat" | Atropos environments ARE verifier harnesses | Harness is spec'd separately |
| "Beat frontier on narrow tasks" | DRQ showed adversarial self-play beats static optimization | No adversarial loop in plans |
| "Weekly refresh cycle" | Atropos supports continuous training | Plans assume manual/offline |

**Bottom line:** The HYDRA plans describe a great product but use an implicit "train it somehow" approach instead of building on:
1. **Atropos** - the RL environment framework already in this repo
2. **DRQ** - the adversarial co-evolution insight that sparked this project

---

## DRQ Core Insight (Why It Matters for HYDRA)

From the Digital Red Queen paper:

> "Adversarial self-play produces more robust strategies than optimization against static benchmarks, even with smaller models."

For HYDRA security training, this means:

| Static Benchmark Approach | DRQ Adversarial Approach |
|--------------------------|-------------------------|
| Model learns to pass fixed test set | Model learns to beat evolving adversary |
| Overfits to benchmark | Generalizes to novel attacks |
| Plateaus when benchmark is "solved" | Continuous improvement arms race |
| Hidden set is only defense against gaming | Gaming becomes the training signal |

**Key DRQ mechanisms that apply:**

1. **Population-based evolution** - Maintain a pool of strategies, not just one model
2. **Relative fitness** - Score against opponents, not absolute metrics
3. **Historical adversaries** - New generations must beat frozen history
4. **Emergent complexity** - Arms race drives capability without manual curriculum

---

## Atropos Architecture (What's Already Here)

From `atroposlib/envs/README.md`:

```
┌─────────────────────────────────────────────────────────────┐
│                    ATROPOS ARCHITECTURE                      │
│                                                              │
│  Environment Microservices (many)                            │
│  ├── Generate rollouts async                                 │
│  ├── Score trajectories                                      │
│  └── Send to Atropos API server                             │
│                    │                                         │
│                    ▼                                         │
│  Atropos API Server (one)                                    │
│  ├── Sequesters rollout data                                │
│  └── Serves batches to trainer                              │
│                    │                                         │
│                    ▼                                         │
│  Trainer (GRPO/DPO/PPO)                                      │
│  └── Backpropagates on batches                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Atropos abstractions:**

| Concept | Purpose | HYDRA Application |
|---------|---------|-------------------|
| `BaseEnv` | Environment microservice base class | Each security task type is an env |
| `collect_trajectories` | Generate rollout data + scores | Run patch/exploit cycle, return scores |
| `ScoredDataGroup` | Batch of rollouts for training | Multiple patch attempts per task |
| `ManagedServer` | Inference + token/logprob tracking | Call models, track outputs |
| Checkpointing | Save/restore evolution state | Store adversary history |

---

## The Integration: HYDRA as Atropos Environment

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       HYDRA + DRQ + ATROPOS                                   │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                    HYDRA ENVIRONMENT (BaseEnv)                         │   │
│  │                                                                        │   │
│  │  Task Pool                    Adversary Pool                           │   │
│  │  ┌─────────────┐             ┌─────────────────────────┐              │   │
│  │  │ sqli-001    │             │ Exploit variants        │              │   │
│  │  │ sqli-002    │             │ (historical + generated)│              │   │
│  │  │ xss-001     │             │                         │              │   │
│  │  │ ...         │             │ Bypass attempts         │              │   │
│  │  └─────────────┘             │ (against "fixed" patches)│             │   │
│  │        │                     └─────────────────────────┘              │   │
│  │        │                              │                                │   │
│  │        └──────────┬───────────────────┘                                │   │
│  │                   ▼                                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │                 VERIFIER HARNESS (from Week1 spec)               │  │   │
│  │  │                                                                  │  │   │
│  │  │  Docker sandbox + Semgrep + Tests + Exploit verification        │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                   │                                                    │   │
│  │                   ▼                                                    │   │
│  │  ┌────────────────────────────────────────────────────────────────┐   │   │
│  │  │                 SCORING (DRQ-style relative fitness)            │   │   │
│  │  │                                                                 │   │   │
│  │  │  Defender score = f(exploit_blocked, tests_pass, scanner_clean) │   │   │
│  │  │  Attacker score = f(bypass_found, under_sandbox_constraints)    │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                   │                                                    │   │
│  │                   ▼                                                    │   │
│  │              ScoredDataGroup → Atropos API → GRPO/DPO trainer         │   │
│  │                                                                        │   │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Two DRQ Forms for HYDRA

**Form 1: Single Specialist vs Evolving Adversary Generator**
- Train: Defender model (patch generator)
- Fixed: Attacker (heuristic exploit mutator or frontier model)
- Simpler to implement, faster iteration
- Good for MVP

**Form 2: Full Adversarial Co-Training**
- Train: Both defender and attacker models
- True arms race like DRQ Core War
- More complex but potentially stronger
- Good for Phase 2+

---

## Concrete Implementation: SecureCodeReviewEnv

Based on the Diplomacy environment pattern (`diplomacy_env_minimal.py`):

```python
"""
SecureCodeReviewEnv: HYDRA security training on Atropos

This environment implements DRQ-style adversarial training for secure code review:
- Defender generates patches for vulnerable code
- Attacker (fixed or trained) generates exploit variants and bypass attempts
- Verifier harness (from Week1 spec) provides deterministic fitness
- Population of historical adversaries prevents overfitting
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from hydra.harness.runner import BenchmarkRunner, TaskResult


class SecureCodeReviewEnvConfig(BaseEnvConfig):
    """Configuration for the secure code review environment."""

    env_name: str = "secure_code_review"

    # Task settings
    tasks_dir: str = "./tasks"
    sandbox_image: str = "hydra-sandbox:latest"

    # DRQ settings
    adversary_pool_size: int = 10  # Historical adversaries to maintain
    attacker_attempts_per_patch: int = 3  # Bypass attempts per patch
    use_trained_attacker: bool = False  # Form 1 vs Form 2

    # Scoring weights
    exploit_blocked_weight: float = 1.0
    tests_passed_weight: float = 0.5
    scanner_clean_weight: float = 0.3
    bypass_found_penalty: float = -2.0  # If attacker finds bypass

    # Training
    total_steps: int = 10000
    steps_per_eval: int = 100


class SecureCodeReviewEnv(BaseEnv):
    """
    HYDRA Secure Code Review Environment for Atropos.

    Implements DRQ-style adversarial training:
    1. Sample task from pool
    2. Defender (training policy) generates patch
    3. Run verifier harness (exploit, tests, scanner)
    4. Attacker generates bypass attempts against "fixed" patch
    5. Score based on defender surviving attacker + harness checks
    6. Add successful attacks to adversary pool
    """

    name = "secure_code_review"
    env_config_cls = SecureCodeReviewEnvConfig

    def __init__(
        self,
        config: SecureCodeReviewEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: SecureCodeReviewEnvConfig = config

        # Initialize harness from Week1 spec
        self.runner = BenchmarkRunner(
            tasks_dir=config.tasks_dir,
            sandbox_image=config.sandbox_image
        )

        # Task pool
        self.task_manifest = self._load_task_manifest()
        self.task_ids = list(self.task_manifest["tasks"].keys())

        # DRQ: Adversary pool (historical attacks that beat previous patches)
        self.adversary_pool: Dict[str, List[Dict]] = {
            task_id: [] for task_id in self.task_ids
        }

        # System prompts
        self.defender_prompt = """You are a security engineer fixing vulnerable code.
Given vulnerable code and a vulnerability description, generate a minimal patch that:
1. Fixes the vulnerability completely
2. Preserves all existing functionality
3. Passes all existing tests

Output only the patched code, no explanation."""

        self.attacker_prompt = """You are a security researcher testing patches.
Given "fixed" code that was patched for a vulnerability, generate a bypass:
1. An exploit that still works despite the patch
2. A mutation of the original attack that evades the fix

Output only the exploit code, no explanation."""

    def _load_task_manifest(self) -> Dict:
        """Load task manifest from tasks directory."""
        manifest_path = Path(self.config.tasks_dir) / "task_manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    @classmethod
    def config_init(cls) -> Tuple[SecureCodeReviewEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = SecureCodeReviewEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
            group_size=4,  # 4 patch attempts per task
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096,
            wandb_name=cls.name,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Set up the environment."""
        # Verify harness is ready
        assert Path(self.config.tasks_dir).exists(), f"Tasks dir not found: {self.config.tasks_dir}"

        # Load any checkpointed adversary pool
        self.load_checkpoint()

    async def get_next_item(self) -> Item:
        """Sample next task for training."""
        import random
        task_id = random.choice(self.task_ids)

        # Load task details
        task_dir = Path(self.config.tasks_dir) / task_id
        with open(task_dir / "task.json") as f:
            task_config = json.load(f)

        # Load vulnerable code
        vuln_file = task_dir / "vulnerable" / task_config["vulnerable_file"]
        with open(vuln_file) as f:
            vulnerable_code = f.read()

        return {
            "task_id": task_id,
            "vulnerable_code": vulnerable_code,
            "vulnerability_description": task_config["description"],
            "cwe": task_config.get("cwe", "unknown"),
            # Include historical adversaries for this task (DRQ)
            "adversary_history": self.adversary_pool.get(task_id, [])[-self.config.adversary_pool_size:],
        }

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """
        Run DRQ-style training loop:
        1. Generate group_size patches (defender)
        2. Verify each patch with harness
        3. Run attacker against patches that pass harness
        4. Score based on surviving both harness AND attacker
        """
        task_id = item["task_id"]
        vulnerable_code = item["vulnerable_code"]
        vuln_description = item["vulnerability_description"]
        adversary_history = item.get("adversary_history", [])

        # Build defender prompt
        messages = [
            {"role": "system", "content": self.defender_prompt},
            {"role": "user", "content": f"""Vulnerable code:
```python
{vulnerable_code}
```

Vulnerability: {vuln_description}

Generate the patched code:"""},
        ]

        # Generate patches with training policy
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=2048,
                temperature=0.7,
            )
            state = managed.get_state()
            nodes = state["nodes"]

        scored_items = []

        for i, (choice, node) in enumerate(zip(completion.choices, nodes)):
            patch_code = choice.message.content.strip()

            # Clean up code blocks if present
            if patch_code.startswith("```"):
                lines = patch_code.split("\n")
                patch_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Step 1: Run harness verification
            harness_result = await self._run_harness(task_id, patch_code)

            # Step 2: Run attacker (DRQ component)
            attacker_score = 0.0
            if harness_result.passed:
                # Only attack patches that pass harness
                attacker_score = await self._run_attacker(
                    task_id, patch_code, vuln_description, adversary_history
                )

            # Step 3: Calculate final score
            score = self._calculate_score(harness_result, attacker_score)

            # Build training data
            full_messages = messages + [{"role": "assistant", "content": patch_code}]

            scored_items.append(ScoredDataItem(
                messages=full_messages if self.config.include_messages else None,
                tokens=node.tokens,
                masks=node.masked_tokens,
                scores=score,
            ))

        # Build ScoredDataGroup
        sdg = ScoredDataGroup(
            tokens=[item["tokens"] for item in scored_items],
            masks=[item["masks"] for item in scored_items],
            scores=[item["scores"] for item in scored_items],
            messages=[item["messages"] for item in scored_items] if self.config.include_messages else None,
        )

        return sdg, []

    async def _run_harness(self, task_id: str, patch_code: str) -> TaskResult:
        """Run Week1 harness on a patch."""
        # This calls the existing harness from 04_BUILD_THIS_Week1_Harness.md
        return await asyncio.to_thread(
            self.runner.run_task, task_id, patch_code
        )

    async def _run_attacker(
        self,
        task_id: str,
        patch_code: str,
        vuln_description: str,
        adversary_history: List[Dict]
    ) -> float:
        """
        DRQ attacker component: Try to bypass the patch.

        Returns negative score if bypass found, 0 if patch holds.
        """
        if not self.config.use_trained_attacker:
            # Form 1: Use fixed attacker (frontier or heuristic)
            return await self._run_fixed_attacker(task_id, patch_code, vuln_description)
        else:
            # Form 2: Use trained attacker (full DRQ)
            return await self._run_trained_attacker(task_id, patch_code, adversary_history)

    async def _run_fixed_attacker(
        self,
        task_id: str,
        patch_code: str,
        vuln_description: str
    ) -> float:
        """
        Form 1: Fixed attacker using frontier model or heuristics.

        This is simpler and good for MVP.
        """
        # Use a secondary server (frontier) for attack generation
        if len(self.servers) < 2:
            # No separate attacker server, use heuristic mutations
            return await self._run_heuristic_attacker(task_id, patch_code)

        attacker_server = self.servers[1]

        attack_prompt = f"""The following code was patched to fix: {vuln_description}

Patched code:
```python
{patch_code}
```

Generate {self.config.attacker_attempts_per_patch} different bypass exploits that might still work.
For each, output just the exploit code."""

        response = await attacker_server.chat_completion(
            messages=[
                {"role": "system", "content": self.attacker_prompt},
                {"role": "user", "content": attack_prompt},
            ],
            n=1,
            max_tokens=1024,
            temperature=0.9,
        )

        # Try each generated exploit
        exploits = self._parse_exploits(response.choices[0].message.content)
        for exploit in exploits:
            success = await self._try_exploit(task_id, patch_code, exploit)
            if success:
                # Bypass found - add to adversary pool
                self._add_to_adversary_pool(task_id, exploit)
                return self.config.bypass_found_penalty

        return 0.0  # Patch survived

    async def _run_heuristic_attacker(self, task_id: str, patch_code: str) -> float:
        """Simple heuristic attack mutations (no model needed)."""
        task_dir = Path(self.config.tasks_dir) / task_id

        # Load original exploit
        with open(task_dir / "exploit" / "exploit.py") as f:
            original_exploit = f.read()

        # Try simple mutations
        mutations = self._generate_exploit_mutations(original_exploit)

        for mutated in mutations:
            success = await self._try_exploit(task_id, patch_code, mutated)
            if success:
                self._add_to_adversary_pool(task_id, {"exploit": mutated, "source": "heuristic"})
                return self.config.bypass_found_penalty

        return 0.0

    def _generate_exploit_mutations(self, exploit: str) -> List[str]:
        """Generate simple exploit mutations."""
        mutations = []

        # Unicode variations
        mutations.append(exploit.replace("'", "\u2019"))  # Smart quote

        # Case variations
        mutations.append(exploit.replace("OR", "oR"))
        mutations.append(exploit.replace("SELECT", "SeLeCt"))

        # Encoding
        mutations.append(exploit.replace("'", "%27"))

        # Comment injection
        mutations.append(exploit.replace("--", "/**/"))

        return mutations[:self.config.attacker_attempts_per_patch]

    async def _try_exploit(self, task_id: str, patch_code: str, exploit: str) -> bool:
        """Try an exploit against patched code."""
        # Write exploit to temp file and run in sandbox
        # Returns True if exploit succeeded (patch failed)
        return await asyncio.to_thread(
            self.runner._run_exploit_with_code,
            patch_code,
            exploit
        )

    async def _run_trained_attacker(
        self,
        task_id: str,
        patch_code: str,
        adversary_history: List[Dict]
    ) -> float:
        """
        Form 2: Trained attacker (full DRQ).

        This requires maintaining a separate attacker model that's also
        being trained. More complex but potentially stronger.
        """
        # TODO: Implement Form 2 after Form 1 is proven
        raise NotImplementedError("Form 2 (trained attacker) not yet implemented")

    def _add_to_adversary_pool(self, task_id: str, adversary: Dict):
        """Add successful attack to DRQ adversary pool."""
        if task_id not in self.adversary_pool:
            self.adversary_pool[task_id] = []

        self.adversary_pool[task_id].append(adversary)

        # Trim to max size
        if len(self.adversary_pool[task_id]) > self.config.adversary_pool_size:
            self.adversary_pool[task_id] = self.adversary_pool[task_id][-self.config.adversary_pool_size:]

    def _calculate_score(self, harness_result: TaskResult, attacker_score: float) -> float:
        """
        Calculate final score combining harness + DRQ attacker.

        This is where DRQ differs from static benchmarks:
        - Static: score = harness_passed (binary)
        - DRQ: score = harness_score + attacker_survival_score (continuous)
        """
        score = 0.0

        if harness_result.passed:
            # Base score for passing harness
            score += self.config.exploit_blocked_weight

            if harness_result.tests_passed:
                score += self.config.tests_passed_weight

            if harness_result.scanner_clean:
                score += self.config.scanner_clean_weight

        # DRQ: Penalty/bonus from attacker
        score += attacker_score

        return score

    def _parse_exploits(self, response: str) -> List[str]:
        """Parse multiple exploits from attacker response."""
        exploits = []
        # Simple parsing - split on code blocks
        parts = response.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd indices are code blocks
                exploits.append(part.strip())
        return exploits[:self.config.attacker_attempts_per_patch]

    def save_checkpoint(self, step: int, data: Dict = None):
        """Save adversary pool to checkpoint."""
        if data is None:
            data = {}
        data["adversary_pool"] = self.adversary_pool
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load adversary pool from checkpoint."""
        super().load_checkpoint()
        if hasattr(self, "adversary_pool_loaded"):
            self.adversary_pool = self.adversary_pool_loaded

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on held-out tasks."""
        # Use hidden set for evaluation
        # Track: pass rate, bypass rate, score distribution
        pass

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log DRQ-specific metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log adversary pool stats
        total_adversaries = sum(len(v) for v in self.adversary_pool.values())
        wandb_metrics[f"{self.name}/adversary_pool_size"] = total_adversaries

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    SecureCodeReviewEnv.cli()
```

---

## How This Changes the HYDRA Plans

### Before (Implicit Training)

```
01_HYDRA_Vision: "Train via DPO/GRPO refresh" (no details)
03_MVP_Path: "First specialist training" (assumes external trainer)
04_BUILD_THIS: "Harness produces JSON" (standalone, not connected to training)
```

### After (Atropos Integration)

```
05_DRQ_Atropos: Harness IS an Atropos environment
                Training IS collect_trajectories → API → GRPO
                DRQ adversarial loop IS built into scoring
```

### What Changes in Existing Docs

| Document | Change Needed |
|----------|--------------|
| 01_HYDRA_Vision | Add section: "Training Platform: Atropos" |
| 03_MVP_Path | Update: harness → Atropos env, training → `python secure_code_review_env.py serve` |
| 04_BUILD_THIS | Keep harness spec, add: "Integration with SecureCodeReviewEnv" |

---

## Implementation Phases

### Phase 0 (Week 1-2): Bridge Week1 Harness to Atropos

**Goal:** Make `04_BUILD_THIS_Week1_Harness.md` work as an Atropos environment.

```yaml
week_1:
  day_1_2:
    - Create SecureCodeReviewEnv skeleton (this document)
    - Import existing harness.runner as subprocess
    - Verify: env.collect_trajectories produces valid ScoredDataGroup

  day_3_4:
    - Connect to Atropos API server
    - Run single task end-to-end: env → API → trainer
    - Verify: GRPO trainer receives batches

  day_5:
    - Add heuristic attacker (Form 1 DRQ, no model)
    - Verify: adversary_pool populates on successful attacks

week_2:
  day_1_2:
    - Scale to 10 tasks (MVP benchmark)
    - Add wandb logging for DRQ metrics
    - Verify: training loop runs 24h without crash

  day_3_5:
    - Add frontier attacker (Form 1 with model)
    - Collect baselines: frontier defender vs heuristic attacker
    - Milestone: DRQ loop proven with 1 training run
```

### Phase 1 (Week 3-4): DRQ Form 1 Validation

**Goal:** Show DRQ Form 1 (fixed attacker) improves specialist.

```yaml
metrics:
  - Specialist pass rate on primary set
  - Specialist survival rate vs attacker
  - Adversary pool growth rate
  - Score improvement over training

success_criteria:
  - Pass rate: >= 80% (same as static benchmark)
  - Survival rate: >= 70% (attacker can't bypass most fixes)
  - Improvement: +10% from baseline over 1000 steps
```

### Phase 2 (Week 5+): DRQ Form 2 (Optional)

**Goal:** True adversarial co-training.

```yaml
form_2_adds:
  - Attacker model (separate from defender)
  - Alternating training: defender step, attacker step
  - Relative Elo scoring instead of absolute metrics
  - Curriculum: start easy, increase attacker strength

form_2_requires:
  - Form 1 proven to work
  - 2x compute (training two models)
  - More complex orchestration
```

---

## Running HYDRA Training

### Before (Implicit)

"Train the model somehow with DPO/GRPO on the verifier output."

### After (Concrete)

```bash
# Terminal 1: Start Atropos API server
cd /home/bron/projects/atropos
python -m atroposlib.cli.run_api --port 8000

# Terminal 2: Start inference server (vLLM/SGLang)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 9004

# Terminal 3: Start HYDRA environment
cd /home/bron/projects/atropos/environments/hydra
python secure_code_review_env.py serve \
    --config config.yaml \
    --rollout-server-url http://localhost:8000

# Terminal 4: Start trainer (GRPO)
cd /home/bron/projects/atropos
python example_trainer/grpo.py \
    --api-url http://localhost:8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir checkpoints/hydra_specialist_v1
```

---

## RelayOne Integration (HYDRA MVP)

### Goals

HYDRA uses RelayOne as the orchestration + telemetry plane for training episodes:

| Layer | RelayOne Surface | HYDRA Use |
|-------|-----------------|-----------|
| **Invocation** | Gateway API | Route verifier/attacker tools through governed endpoints |
| **Dataset mining** | Audit logs | Export tamper-evident logs as training corpora |
| **Visualization** | OpenTelemetry | Record spans per episode, browse traces |
| **Deployment** | Model routing | Register trained specialists into routing rules |

This makes training look like production: governed calls, structured logs, and traceability.

### Phase 0 (MVP): Demo Gateway Endpoint

For Week 1–2, HYDRA uses the no-auth demo invoke endpoint to remove auth setup friction.

**Endpoint:**
```
POST /gateway/demo/agents/:agentId/invoke
```

**Defined in:** `mvp/apps/api/src/routes/gateway.ts`

**Request body:**
```json
{
  "tool": "string",
  "input": { "any": "json" },
  "timeout": 30000,
  "metadata": { "episode_id": "...", "task_id": "...", "role": "defender|attacker" }
}
```

**How HYDRA uses this in `SecureCodeReviewEnv.collect_trajectories()`:**

1. **Defender step:** Generate patch candidates, then call RelayOne verifier agent:
   - `tool: "verify_patch"`
   - `input: { task_id, patch, ... }`

2. **Attacker step (DRQ):** Generate bypass attempts, then call:
   - `tool: "try_bypass"`
   - `input: { task_id, patch, bypass_payload, ... }`

RelayOne becomes the "arena API" while Atropos remains the RL data engine.

### RelayOne Local Dev Mode

RelayOne provides a local dev mode where auth is disabled and demo endpoints are enabled.

**Reference:**
- `mvp/docs/guides/local-dev-mode.md`
- `mvp/deploy/docker/docker-compose.devmode.yml`

**Quick start:**
```bash
cd mvp/deploy/docker
docker-compose -f docker-compose.devmode.yml up
```

**Access points:**
- RelayOne API base URL (devmode compose): `http://localhost:3000`
  - Note: devmode sets `API_PORT=3000`; app.ts uses `PORT || 3001`. Port mapping in compose handles this.
- Env toggles (devmode defaults):
  - `DISABLE_AUTH=true`
  - `DISABLE_DEMO_ENDPOINTS=false`
  - `RELAY_DEV_MODE=true`

### Dataset Mining: Audit Logs → Training Data

RelayOne includes a comprehensive audit log system with integrity chaining (HMAC).

**Code:**
- `mvp/apps/api/src/services/audit.service.ts`
- `mvp/apps/api/src/routes/audit.ts`

> **⚠️ Registration required:** As of this writing, `auditRoutes` is defined but not registered in `app.ts`. To enable audit export, add:
> ```typescript
> // In mvp/apps/api/src/app.ts
> import { auditRoutes } from './routes/audit';
> fastify.register(auditRoutes, { prefix: '/api/v1/audit' });
> ```

**Export endpoint (when registered):**
```
POST /api/v1/audit/export
```

Use this to export:
- Successful `verify_patch` runs (positive examples)
- Failures with structured reasons (negative examples / preference pairs)
- Policy blocks (PII/ACL/HITL later)

**Metadata convention for mineable logs:**

When calling the gateway, include stable identifiers in `metadata`:
```json
{
  "episode_id": "ep-001",
  "task_id": "sqli-001",
  "candidate_id": "patch-42",
  "role": "defender|attacker|verifier",
  "model_version": "hydra-v1.0"
}
```

### Episode Visualization: OpenTelemetry Traces

RelayOne exposes OTel span management endpoints (registered at both `/otel` and `/api/v1/otel`).

**Code:** `mvp/apps/api/src/routes/otel.ts`

**Endpoints (both paths work):**
```
POST /otel/spans/start       # or /api/v1/otel/spans/start
POST /otel/spans/end         # or /api/v1/otel/spans/end
POST /otel/spans/record      # or /api/v1/otel/spans/record
```

**Recommended span types:**
- `agent.run` (episode-level)
- `tool.call` (verifier invocation)
- `model.call` (LLM inference)
- `policy.check` (optional)
- `hitl.step` (later phase)

**Recommended attributes:**
```
hydra.episode_id
hydra.task_id
hydra.role
hydra.score
hydra.harness_passed
hydra.attacker_bypassed
model.name
model.provider
```

This makes it possible to view training like production incident traces.

### Phase 1+: Authenticated Gateway + Full Governance

Once the harness loop is stable, switch from demo invoke to authenticated invoke:

```
POST /gateway/agents/:agentId/invoke
```

This enables "governed-from-day-1" training:
- ACL firewall checks
- PII detection blocks
- Consent enforcement
- HITL approvals
- Billing/spending policy gates

(Implemented in `mvp/apps/api/src/services/gateway.service.ts`)

### Deployment Loop: Register HYDRA Specialists into RelayOne Routing

RelayOne contains model routing layers:

| Layer | Code Location |
|-------|--------------|
| Node routing | `mvp/apps/api/src/services/llm-router.service.ts` |
| Rust routing | `mvp/apps/gateway-rust/src/services/model_router/mod.rs` |
| Semantic cache | `mvp/apps/gateway-rust/src/services/semantic_cache/*` |

**Deployment plan:**

1. Register trained HYDRA specialist as on-prem/self-hosted provider endpoint
2. Route task category (e.g. `secure_code_review`) to HYDRA specialist first
3. Keep fallback chain to frontier models for low-confidence requests
4. Use inference logs + audit logs to mine new failure cases for refresh cycles

---

## DRQ Metrics to Track

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| `harness_pass_rate` | Basic patch quality | >= 80% |
| `attacker_survival_rate` | Robustness to bypasses | >= 70% |
| `adversary_pool_growth` | How often attacker wins | Decreasing over time |
| `score_mean` | Overall training signal | Increasing over time |
| `score_variance` | Exploration vs exploitation | Moderate (not too low) |

---

## Comparison: Static vs DRQ Training

### Static Benchmark (Current HYDRA Plans)

```python
# What 04_BUILD_THIS does
score = 1.0 if harness.passed else 0.0
# Problem: Model can overfit to fixed test set
```

### DRQ Training (This Integration)

```python
# What 05_DRQ_Atropos does
harness_score = compute_harness_score(result)
attacker_score = run_attacker(patch, adversary_history)
score = harness_score + attacker_score
# Benefit: Model must generalize to novel attacks
```

### Expected Outcome

| Training Type | In-Distribution | Out-of-Distribution |
|--------------|-----------------|---------------------|
| Static benchmark only | High | Medium (overfits) |
| DRQ adversarial | High | Higher (generalizes) |

This is the core DRQ insight: **the arms race drives generalization**.

---

## File Structure Update

> **⚠️ Planned, not yet implemented:** The `environments/hydra/` directory and its files do not exist yet. This structure is the target for Week 2+ implementation.

```
atropos/
├── environments/
│   ├── hydra/                          # PLANNED: HYDRA security environments
│   │   ├── secure_code_review_env.py   # Implement in Week 2
│   │   ├── detection_engineer_env.py   # Future: Sigma/YARA DRQ
│   │   ├── harness/                    # From 04_BUILD_THIS
│   │   │   ├── runner.py
│   │   │   ├── Dockerfile
│   │   │   └── semgrep_rules/
│   │   └── tasks/                      # Benchmark tasks
│   │       ├── task_manifest.json
│   │       ├── sqli-001/
│   │       └── ...
│   └── game_environments/
│       └── diplomacy_environment/      # Reference pattern
│
├── docs/Plans/
│   ├── 01_HYDRA_Vision_and_Architecture.md
│   ├── 02_Security_Governance_Framework.md
│   ├── 03_MVP_Path_to_90.md
│   ├── 04_BUILD_THIS_Week1_Harness.md
│   └── 05_DRQ_Atropos_Integration.md   # THIS DOCUMENT
```

---

## Summary

**What was missing:**
1. HYDRA plans don't use Atropos (the RL framework in this repo)
2. HYDRA plans don't implement DRQ (the original inspiration)
3. Training was assumed to be "some offline DPO/GRPO" without specifics

**What this document adds:**
1. `SecureCodeReviewEnv` - HYDRA harness as Atropos environment
2. DRQ Form 1 - Fixed attacker drives generalization
3. Concrete commands to run training
4. Metrics and phases for validation

**The core change:**
- Before: "Build harness → generate data → train offline"
- After: "Build harness → embed in Atropos env → train online with DRQ adversary"

**Next action:** Implement `SecureCodeReviewEnv` that wraps Week1 harness and adds DRQ attacker loop.

---

*Document Version: 1.0*
*Status: Integration specification*
*Prerequisite: 04_BUILD_THIS_Week1_Harness.md completed*
