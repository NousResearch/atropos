# Master LLM Training Plan: Security Specialist Flywheel

## Project Codename: HYDRA (Hybrid Universal Dynamic Reinforcement Adversaries)

**Version:** 2.0
**Date:** January 2026
**Authors:** AI Research Team

---

## Executive Summary

This plan outlines a practical approach to building **security-focused LLM specialists** that can reliably beat frontier models on narrow, operational tasks. The key insight: you don't beat frontier models by training once—you build a **continuous specialization flywheel**.

**Core Architecture:** CI/CD for Model Advantage
1. Use frontier models as **teachers** (selectively)
2. Generate **verifiable, domain-specific data** that general models don't have
3. Train/refresh specialist models **regularly**
4. Route requests to specialist OR frontier with **eval-driven gating**

**Key Innovation:** The moat isn't "smarter weights"—it's **better verifiers + domain-specific data + faster iteration**.

**Products:**
1. **Detection Engineer Model** - Sigma/YARA rules, alert triage, IR playbooks
2. **Secure Code Review Model** - Vulnerability patches, security review, CI copilot
3. **CTF Operator Model** - Authorized red team assistance (sandboxed)

**Expected Outcome:** Specialists that outperform frontier models on specific security tasks through verifiable, continuous improvement—not one-time training.

**Training Platform:** [Atropos](https://github.com/NousResearch/atropos) - the LLM RL Gym framework in this repository. HYDRA environments are Atropos `BaseEnv` subclasses that generate rollouts for GRPO/DPO training.

**Inspiration:** [Digital Red Queen](https://sakana.ai/drq/) - adversarial co-evolution produces more robust strategies than static benchmark optimization, even with smaller models.

---

## Table of Contents

1. [Continuous Specialization Flywheel](#1-continuous-specialization-flywheel)
2. [Training Platform: Atropos](#2-training-platform-atropos)
3. [Product Verticals](#3-product-verticals)
4. [Verifier Harnesses (The Moat)](#4-verifier-harnesses-the-moat)
5. [DRQ Adversarial Training](#5-drq-adversarial-training)
6. [Two-Tier Model Architecture](#6-two-tier-model-architecture)
7. [Judge Reliability & Bias Mitigation](#7-judge-reliability--bias-mitigation)
8. [Data Evolution Loop](#8-data-evolution-loop)
9. [Implementation Phases](#9-implementation-phases)
10. [Infrastructure & Costs](#10-infrastructure--costs)
11. [Data Specification](#11-data-specification)
12. [Budget Gates & Stopping Criteria](#12-budget-gates--stopping-criteria)
13. [Ablation Plan](#13-ablation-plan)
14. [Risk Analysis](#14-risk-analysis)
15. [Success Metrics](#15-success-metrics)

---

## 1. Continuous Specialization Flywheel

### 1.1 The "Always Stay Ahead" Principle

**Key Insight:** You don't beat frontier models by training once. You beat them by building a continuous specialization flywheel that:

1. Uses the newest frontier models as **teachers** (selectively)
2. Generates **verifiable, domain-specific data** that general models don't have
3. Trains/refreshes specialist models **regularly**
4. Routes requests to specialist OR frontier, with **eval-driven gating**

**Think: "CI/CD for model advantage."**

### 1.2 The Flywheel Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS SPECIALIZATION FLYWHEEL                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FRONTIER TEACHER LAYER                           │   │
│  │  (Claude Opus / GPT-5 / Gemini Pro - via API, NOT trained)          │   │
│  │                                                                      │   │
│  │  Used for:                                                           │   │
│  │  • Generating hard scenarios                                         │   │
│  │  • Adjudicating disagreements                                        │   │
│  │  • Producing high-quality rationales                                 │   │
│  │  • Auditing dataset quality                                          │   │
│  │  • Teacher refresh when new model drops                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     VERIFIER / SIMULATOR LAYER                       │   │
│  │  ████████████████████ THIS IS THE MOAT ████████████████████████      │   │
│  │                                                                      │   │
│  │  • Sandboxed code execution                                          │   │
│  │  • Unit tests / fuzzing                                              │   │
│  │  • Static analyzers (Semgrep, linters, taint checkers)              │   │
│  │  • Containerized vulnerable labs                                     │   │
│  │  • IDS rule tests, log replay, alert matching                       │   │
│  │  • Policy checkers (blue-team constraints)                          │   │
│  │  • Exploit verification in sandboxes                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     SPECIALIST MODEL LAYER                           │   │
│  │  (Your products - trained via LoRA, refreshed regularly)            │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │ Detection   │  │ Secure Code │  │ CTF/Lab     │                  │   │
│  │  │ Engineer    │  │ Reviewer    │  │ Operator    │                  │   │
│  │  │ Model       │  │ Model       │  │ Model       │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ROUTER / ORCHESTRATOR                            │   │
│  │                                                                      │   │
│  │  • Picks specialist vs frontier based on task type + confidence     │   │
│  │  • Falls back to frontier when specialist is uncertain              │   │
│  │  • Logs all interactions for future training                        │   │
│  │  • Collects failure cases for next refresh cycle                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FEEDBACK LOOP (Weekly/Monthly)                   │   │
│  │                                                                      │   │
│  │  1. Mine failures where specialist lost to frontier                 │   │
│  │  2. Ask frontier for better solutions (or critiques)                │   │
│  │  3. Add to dataset + verify                                         │   │
│  │  4. Run preference optimization refresh (DPO/GRPO)                  │   │
│  │  5. Re-evaluate and ship new "vX.Y" specialist                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Production Implementation:** The Router/Orchestrator layer is implemented via **RelayOne** (`mvp/`). See [RelayOne Integration](05_DRQ_Atropos_Integration.md#relayone-integration-hydra-mvp) for gateway APIs, audit log export, and model routing configuration.

### 1.3 Teacher Refresh Protocol

**When a new frontier model drops (e.g., Claude 5, GPT-6):**

```python
class TeacherRefresh:
    """Protocol for refreshing specialists when new frontier drops."""

    async def refresh_cycle(self, new_frontier_model: str):
        # Step 1: Sample hard cases where specialist loses
        hard_cases = await self.mine_failures(
            specialist=self.current_specialist,
            frontier=new_frontier_model,
            n_samples=1000
        )

        # Step 2: Ask new frontier for better solutions
        improved_solutions = await self.generate_with_teacher(
            model=new_frontier_model,
            hard_cases=hard_cases
        )

        # Step 3: Verify solutions (THE MOAT - no judge needed)
        verified = []
        for case, solution in zip(hard_cases, improved_solutions):
            result = await self.verifier.verify(case, solution)
            if result.passed:
                verified.append((case, solution))

        # Step 4: Add to dataset
        self.dataset.add_preference_pairs(verified)

        # Step 5: Run short preference optimization
        new_specialist = await self.train_refresh(
            base=self.current_specialist,
            new_data=verified,
            method="dpo",  # Quick refresh, not full retrain
            epochs=1
        )

        # Step 6: Evaluate and ship
        eval_results = await self.evaluate(new_specialist)
        if eval_results.better_than(self.current_specialist):
            self.deploy(new_specialist, version=self.next_version())
```

### 1.4 Why Specialists Beat Frontier

| Factor | Generic Frontier Model | Your Specialist |
|--------|----------------------|-----------------|
| **Optimization target** | Broad helpfulness + safety | Your exact task metrics |
| **Tool chain training** | Generic tool use | Your specific tools + workflows |
| **Failure modes** | Doesn't know your edge cases | Trained on your failures |
| **Feedback loop** | Updated every few months | Updated weekly |
| **Verification** | Relies on RLHF | Deterministic verifiers |
| **Telemetry formats** | Generic | Your exact log formats |
| **False positive tuning** | Not a priority | Optimized for low FP |

**The moat isn't "smarter weights." It's:**
- Better data (domain-specific, continuously generated)
- Better verifiers (deterministic, no judge bias)
- Tighter task framing (exactly your workflow)
- Faster iteration (weekly refresh vs monthly releases)

---

## 2. Training Platform: Atropos

HYDRA runs on the **Atropos** LLM RL Gym framework (this repository). Each HYDRA product is an Atropos environment that generates training rollouts.

### 2.1 Why Atropos (Not Standalone Harness)

| Standalone Harness | Atropos Environment |
|-------------------|---------------------|
| Produces pass/fail JSON | Produces `ScoredDataGroup` for GRPO/DPO |
| Manual offline training | Online training via API server |
| Human-operated refresh | Continuous automated training |
| Single task at a time | Parallel multi-environment scaling |

### 2.2 Atropos Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYDRA ON ATROPOS                                   │
│                                                                             │
│  HYDRA Environments (BaseEnv subclasses)                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ SecureCodeReview│  │ DetectionEngineer│  │ CTFOperator     │            │
│  │ Env             │  │ Env              │  │ Env             │            │
│  │                 │  │                  │  │                 │            │
│  │ • Task pool     │  │ • Log corpus     │  │ • CTF labs      │            │
│  │ • Exploit verify│  │ • Rule verify    │  │ • Tool traces   │            │
│  │ • DRQ attacker  │  │ • FP analysis    │  │ • Sandbox       │            │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘            │
│           │                    │                     │                      │
│           └────────────────────┼─────────────────────┘                      │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Atropos API Server                                │   │
│  │  • Sequesters rollout data from all environments                    │   │
│  │  • Serves batches to trainer                                        │   │
│  │  • Manages checkpoints                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GRPO/DPO Trainer                                  │   │
│  │  • Fetches batches from API                                         │   │
│  │  • Backpropagates                                                   │   │
│  │  • Produces LoRA checkpoints                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 How Training Actually Works

```bash
# Terminal 1: Start Atropos API server
python -m atroposlib.cli.run_api --port 8000

# Terminal 2: Start inference server (vLLM/SGLang)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 9004

# Terminal 3: Start HYDRA environment (generates rollouts)
python environments/hydra/secure_code_review_env.py serve \
    --rollout-server-url http://localhost:8000

# Terminal 4: Start trainer (consumes rollouts)
python example_trainer/grpo.py \
    --api-url http://localhost:8000 \
    --output-dir checkpoints/hydra_v1
```

### 2.4 HYDRA Environment Pattern

Each HYDRA product follows this pattern (based on `diplomacy_env_minimal.py`):

```python
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup

class SecureCodeReviewEnv(BaseEnv):
    """HYDRA Secure Code Review as Atropos environment."""

    name = "secure_code_review"

    async def get_next_item(self) -> Item:
        """Sample next task from pool."""
        return {"task_id": "sqli-001", "vulnerable_code": "..."}

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List[Item]]:
        """Generate patches, verify with harness, score."""
        # 1. Generate patch candidates with training policy
        patches = await self.generate_patches(item)

        # 2. Verify each patch (exploit, tests, scanner)
        results = [self.runner.run_task(item["task_id"], p) for p in patches]

        # 3. Score and return ScoredDataGroup for GRPO
        return self.build_scored_data_group(patches, results)
```

---

## 3. Product Verticals

### 3.1 Product Strategy

**Focus on narrow, operational security tasks where:**
1. Success is **machine-verifiable** (tests, scanners, log replay)
2. Generic models **don't have your data** (internal formats, edge cases)
3. Tasks are **tool-grounded** (code execution, API calls)
4. There's **clear ROI** (time saved, incidents prevented)

### 3.2 Product 1: Detection Engineer Model (Blue Team)

**Target users:** SOC analysts, detection engineers, threat hunters

**Core tasks:**

| Task | Input | Output | Verifier |
|------|-------|--------|----------|
| Log → Detection Rule | Zeek/Suricata/Sysmon logs | Sigma/YARA/Suricata rules | Replay logs, check triggers |
| Alert Triage | Alert bundle + context | Prioritized hypotheses + steps | Simulated incident with ground truth |
| Rule Optimization | Existing rule + FP examples | Refined rule | FP rate on test corpus |
| IOC Extraction | Threat report | Structured IOCs | Format validation + known-bad matching |

**Dataset generation pipeline:**

```python
class DetectionEngineerDataset:
    """Generate verifiable detection engineering data."""

    async def generate_log_to_rule_pairs(self):
        # 1. Collect logs from diverse sources
        logs = await self.collect_logs(sources=[
            "zeek_samples", "sysmon_samples", "windows_security"
        ])

        # 2. For each attack type, generate rule candidates
        for log_bundle in logs:
            attack_type = self.classify_attack(log_bundle)

            # Ask frontier to generate Sigma rule
            rule_candidate = await self.frontier.generate(
                f"Write a Sigma rule to detect this attack:\n{log_bundle}"
            )

            # VERIFY: Replay logs and check rule triggers
            verification = await self.sigma_verifier.test_rule(
                rule=rule_candidate,
                positive_logs=log_bundle.attack_logs,
                negative_logs=log_bundle.benign_logs
            )

            if verification.true_positive_rate > 0.95 and verification.false_positive_rate < 0.05:
                self.dataset.add({
                    "input": log_bundle,
                    "output": rule_candidate,
                    "verification": verification
                })
```

**Why it beats frontier:**
- Trained on **your exact telemetry formats** and tool chain
- Optimized for **low false positives** (generic models aren't tuned for this)
- Knows **your environment** (asset inventory, baseline behavior)
- **Weekly refresh** on new attack patterns

### 3.3 Product 2: Secure Code Review / Patch Engineer

**Target users:** Security engineers, DevSecOps, AppSec teams

**Core tasks:**

| Task | Input | Output | Verifier |
|------|-------|--------|----------|
| Vulnerability Patch | Code + scan finding + exploit | Patched code + tests | Scanner clean + tests pass + exploit fails |
| Security Review | PR diff | Security issues + fixes | Static analyzer + manual spot check |
| Secure Refactor | Insecure pattern | Secure equivalent | Regression tests + security tests |
| Dependency Fix | CVE + affected code | Updated deps + migration | Build passes + scanner clean |

**Dataset generation pipeline:**

```python
class SecureCodeReviewDataset:
    """Generate verifiable secure code review data."""

    async def generate_patch_pairs(self):
        # 1. Find vulnerable code samples
        vuln_samples = await self.collect_vulnerable_code(sources=[
            "cve_fixes",  # Historical CVE patches
            "sast_findings",  # Static analysis findings
            "ctf_challenges"  # CTF web challenges
        ])

        for sample in vuln_samples:
            # 2. Ask frontier to generate patch
            patch = await self.frontier.generate(f"""
                Vulnerable code: {sample.vulnerable_code}
                Vulnerability: {sample.vulnerability_description}
                Generate: Patched code + test cases
            """)

            # 3. VERIFY: Multi-stage verification
            verification = await self.verify_patch(
                original=sample.vulnerable_code,
                patched=patch.code,
                tests=patch.tests
            )

            if verification.all_passed:
                self.dataset.add({
                    "input": sample,
                    "output": patch,
                    "verification": verification
                })

    async def verify_patch(self, original, patched, tests):
        results = {}
        results["exploit_original"] = await self.sandbox.run_exploit(original)
        results["exploit_patched"] = await self.sandbox.run_exploit(patched)
        results["regression"] = await self.sandbox.run_tests(patched, tests.regression)
        results["scanner"] = await self.semgrep.scan(patched)

        return VerificationResult(
            all_passed=(
                results["exploit_original"].succeeded and
                not results["exploit_patched"].succeeded and
                results["regression"].all_passed and
                results["scanner"].findings == 0
            )
        )
```

**Why it beats frontier:**
- **Deterministic evaluation** - you can grind improvement
- Trained on **your stack** (frameworks, libraries, patterns)
- Becomes a **CI security copilot** that's objectively better
- Very strong ROI (vulnerabilities caught in CI vs production)

### 3.4 Product 3: CTF/Lab Operator Model (Authorized Red Team)

**Target users:** Pentest teams, red team operators, security trainers

**Constraints (critical for safety):**
- Constrained to **CTF-style labs you own**
- Requires **tool traces + reproducibility + auditing**
- Outputs **non-actionable outside sandbox**
- All executions **logged and reviewable**

**Core tasks:**

| Task | Input | Output | Verifier |
|------|-------|--------|----------|
| Attack Chain Planning | Lab description + objective | Step-by-step plan | Successful completion in lab |
| Exploit Development | Vulnerability description | Working exploit | Exploit succeeds in sandbox |
| Lateral Movement | Current access + target | Next steps + rationale | Access achieved in lab |

**Safety notes:**
- This product requires **strong governance**
- Only viable for authorized testing companies
- Audit logs are **non-negotiable**

### 3.5 Product Comparison

| Product | Controversy Risk | Sellability | Verification Complexity | ROI |
|---------|------------------|-------------|------------------------|-----|
| Detection Engineer | **LOW** | **HIGH** | Medium | Very High |
| Secure Code Review | **LOW** | **HIGH** | Medium | Very High |
| CTF Operator | **MEDIUM** | Medium | High | High |

**Recommendation: Start with Detection Engineer or Secure Code Review.**

---

## 4. Verifier Harnesses (The Moat)

### 4.1 The Single Most Important Design Rule

**Make your domain tasks machine-checkable.**

If success depends on a **judge model**, your advantage gets eaten as soon as the next frontier model has better judging.

If success depends on **tests, sandboxes, log replays, scanners**, you can keep compounding improvements cheaply.

### 4.2 Verifier Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VERIFIER HARNESS STACK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 1: Code Execution Sandbox                                        │
│  ├── Docker containers with strict isolation                            │
│  ├── Firecracker microVMs for untrusted code                           │
│  ├── Resource limits (CPU, memory, time, network)                       │
│  └── Reproducible environments (pinned dependencies)                    │
│                                                                         │
│  LAYER 2: Static Analysis                                               │
│  ├── Semgrep (custom rules for your patterns)                          │
│  ├── CodeQL (for complex taint analysis)                               │
│  ├── Bandit/Pylint/ESLint (language-specific)                          │
│  └── Custom AST checks                                                  │
│                                                                         │
│  LAYER 3: Dynamic Testing                                               │
│  ├── Unit test execution                                                │
│  ├── Fuzzing (AFL++, libFuzzer)                                        │
│  ├── Property-based testing (Hypothesis)                               │
│  └── Exploit verification in sandbox                                   │
│                                                                         │
│  LAYER 4: Security-Specific                                             │
│  ├── Sigma rule testing (against log corpora)                          │
│  ├── YARA rule testing (against file corpora)                          │
│  ├── Suricata/Snort rule testing (against PCAP)                        │
│  └── IDS alert matching and false positive analysis                    │
│                                                                         │
│  LAYER 5: Environment Simulation                                        │
│  ├── Vulnerable lab environments (VulnHub, HackTheBox-style)           │
│  ├── Log replay systems                                                 │
│  ├── Simulated incident environments                                   │
│  └── Network traffic replay (tcpreplay)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Verifier Coverage by Product

| Product | Primary Verifiers | Secondary Verifiers |
|---------|------------------|---------------------|
| Detection Engineer | Sigma/YARA test, log replay | FP rate analysis, format validation |
| Secure Code Review | Static analysis, exploit sandbox | Unit tests, fuzz testing |
| CTF Operator | Lab execution, objective check | Audit log, safety bounds |

---

## 5. DRQ Adversarial Training

### 5.1 Digital Red Queen Insight

From [Sakana AI's DRQ paper](https://sakana.ai/drq/):

> "Adversarial self-play produces more robust strategies than optimization against static benchmarks, even with smaller models."

**Why this matters for HYDRA:**

| Static Benchmark Training | DRQ Adversarial Training |
|--------------------------|-------------------------|
| Model learns to pass fixed test set | Model learns to beat evolving adversary |
| Overfits to benchmark | Generalizes to novel attacks |
| Plateaus when benchmark is "solved" | Arms race drives continuous improvement |
| Hidden set is only defense against gaming | Gaming becomes the training signal |

### 5.2 DRQ for Security Tasks

HYDRA naturally has **adversarial pairs**:

| Defender | Attacker | Arena |
|----------|----------|-------|
| Patch generator | Exploit mutator | Secure Code Review |
| Detection rule writer | Evasion generator | Detection Engineer |
| Defender playbook | Attack chain planner | CTF Operator |

**DRQ Form 1 (MVP):** Fixed attacker
- Train defender only
- Attacker is heuristic or frontier model
- Simpler, proves the loop works

**DRQ Form 2 (Advanced):** Full co-training
- Train both defender and attacker
- True arms race like Sakana's Core War
- More complex but potentially stronger

### 5.3 DRQ in Atropos collect_trajectories

```python
async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List[Item]]:
    """DRQ-style scoring in Atropos environment."""

    # 1. Generate patches (defender)
    patches = await self.generate_patches(item)

    # 2. Verify with harness (static benchmark component)
    harness_results = [self.runner.run_task(item["task_id"], p) for p in patches]

    # 3. Run attacker against patches that pass harness (DRQ component)
    attacker_results = []
    for patch, result in zip(patches, harness_results):
        if result.passed:
            bypass = await self.run_attacker(patch, item)
            attacker_results.append(bypass)
        else:
            attacker_results.append(None)

    # 4. Score combines harness + attacker survival
    scores = [
        self.compute_drq_score(harness, attack)
        for harness, attack in zip(harness_results, attacker_results)
    ]

    return self.build_scored_data_group(patches, scores)
```

### 5.4 Adversary Pool (Historical Opponents)

DRQ's key mechanism: **new generations must beat historical opponents**.

```python
class AdversaryPool:
    """Store successful attacks for future training."""

    def __init__(self, max_size: int = 100):
        self.pool: Dict[str, List[Dict]] = {}  # task_id -> attacks

    def add_successful_attack(self, task_id: str, attack: Dict):
        """When attacker bypasses a patch, store it."""
        if task_id not in self.pool:
            self.pool[task_id] = []
        self.pool[task_id].append(attack)
        # Trim to max size
        self.pool[task_id] = self.pool[task_id][-self.max_size:]

    def get_historical_attacks(self, task_id: str) -> List[Dict]:
        """Get attacks that future patches must survive."""
        return self.pool.get(task_id, [])
```

This prevents overfitting: patches must work against all historical attacks, not just current ones.

---

## 6. Two-Tier Model Architecture

### 6.1 Tier Separation

**Tier 1 (Teachers/Orchestrators):** Frontier models via API
- Used for: data generation, adjudication, auditing
- NOT trained, just used
- Examples: Claude Opus 4.5, GPT-5.2, Gemini 3 Pro

**Tier 2 (Specialists):** Open-source models we train
- Trained via LoRA for efficiency
- Refreshed regularly (weekly/monthly)
- Examples: Qwen2.5-7B, Llama 3.3-8B, DeepSeek-R1-7B

### 6.2 API Access

| Access Method | Use Case | Cost |
|---------------|----------|------|
| **OpenRouter** | Single API, 300+ models | ~5% markup |
| **Direct APIs** | High volume, lower cost | Provider rates |
| **Claude Code API** | Max plan included | Subscription |

---

## 7. Judge Reliability & Bias Mitigation

### 7.1 Known LLM-as-Judge Biases

| Bias Type | Description | Mitigation |
|-----------|-------------|------------|
| **Position Bias** | Prefers A or B based on order | Always evaluate both orders |
| **Verbosity Bias** | Prefers longer answers | Length normalization |
| **Self-Enhancement** | Prefers its own style | Use different judge than generator |

### 7.2 Required Mitigations

```python
async def mitigated_judge(self, response_a, response_b, challenge):
    """Always evaluate in both orders."""

    verdict_ab = await self.judge.evaluate(first=response_a, second=response_b)
    verdict_ba = await self.judge.evaluate(first=response_b, second=response_a)

    if verdict_ab.winner == "first" and verdict_ba.winner == "second":
        return {"winner": "A", "confident": True}
    elif verdict_ab.winner == "second" and verdict_ba.winner == "first":
        return {"winner": "B", "confident": True}
    else:
        return {"winner": None, "needs_escalation": True}
```

### 7.3 Judge Calibration (Week 0 Requirement)

Before training, verify:
- Position bias flip rate < 10%
- Verbosity correlation < 0.3
- Self-agreement > 95%

---

## 8. Data Evolution Loop

### 8.1 Key Principle

**Treat "evolution" as DATA evolution, not weight evolution.**

Directly mutating LoRA weights is inefficient. Instead:
1. Generate many candidate responses
2. Select winners via verifiers/judges
3. Train using preference optimization (DPO/GRPO)

### 8.2 The Loop

```
Generate k responses → Verify/Judge → Select winners → Train DPO → Repeat
```

### 8.3 Training Objective Per Arena

| Arena | Method | Why |
|-------|--------|-----|
| Code/Security | GRPO | Multiple responses ranked by test results |
| Reasoning | DPO | Binary correct/incorrect |
| Subjective | KTO/ORPO | Handles noisy preferences |

---

## 9. Implementation Phases

### Phase 0: Verifier Harness (Week 0-1)

**Goal:** Build the moat before anything else

**Tasks:**
- [ ] Set up Docker sandbox for code execution
- [ ] Implement Sigma rule tester with log corpus
- [ ] Implement static analyzer integration (Semgrep)
- [ ] Build exploit verification sandbox
- [ ] Create test corpora (attack logs, vulnerable code)

**Compute:** 1x A100 or consumer GPU (~$50)

**GO/NO-GO:** Verifiers must work reliably before Phase 1

### Phase 1: First Specialist (Weeks 1-3)

**Goal:** Prove the loop works on ONE product

**Tasks:**
- [ ] Choose product: Detection Engineer OR Secure Code Review
- [ ] Generate initial dataset via frontier + verification
- [ ] Train first specialist with DPO
- [ ] Set up eval pipeline
- [ ] Deploy behind router with frontier fallback

**Compute:** 2x A100 80GB (~$300)

**Success Criteria:**
- [ ] +5% improvement on domain benchmark
- [ ] Verifier pass rate > 90%

### Phase 2: Router & Refresh (Weeks 3-6)

**Goal:** Implement continuous improvement

**Tasks:**
- [ ] Build router with confidence-based fallback
- [ ] Implement failure mining pipeline
- [ ] Run first teacher refresh cycle
- [ ] Ship v1.1 specialist

**Compute:** 4x A100 + ~$200 API (~$1,200)

### Phase 3: Scale & Second Product (Weeks 6+)

**Goal:** Expand only with proven signal

**Tasks:**
- [ ] Add second product vertical
- [ ] Scale to production workloads
- [ ] Establish weekly refresh cadence

**Compute:** 8x A100 (~$3,000+)

---

## 10. Infrastructure & Costs

### 10.1 Cost-Minimized Compute Ramp

| Phase | GPUs | Duration | Compute | API | Total |
|-------|------|----------|---------|-----|-------|
| Phase 0 | 1x A100 | 1 week | $50 | $0 | $50 |
| Phase 1 | 2x A100 | 2 weeks | $300 | $0 | $300 |
| Phase 2 | 4x A100 | 3 weeks | $1,000 | $200 | $1,200 |
| Phase 3 | 8x A100 | 4 weeks | $3,000 | $300 | $3,300 |
| **Total** | - | 10 weeks | **$4,350** | **$500** | **$4,850** |

### 10.2 Model Selection

**Start with ONE base model:** Qwen2.5-7B-Instruct

**Why:**
- Strong baseline performance
- Efficient inference
- Good LoRA compatibility
- Apache 2.0 license

---

## 11. Data Specification

### 11.1 What Is Stored Per Match

```python
@dataclass
class MatchRecord:
    match_id: str
    timestamp: datetime
    arena_type: str

    # Challenge
    challenge_text: str
    challenge_metadata: Dict

    # Response
    response: str
    response_tokens: int

    # Verification
    verifier_result: Dict
    verification_passed: bool

    # Costs
    inference_cost: float
    total_cost: float
```

### 11.2 Caching Rule

**Never pay twice for the same computation.**

---

## 12. Budget Gates & Stopping Criteria

### 12.1 Hard Gates

| Gate | Trigger | Action |
|------|---------|--------|
| **Week 3** | No +5% on domain metric | STOP or PIVOT |
| **Daily API** | >$50/day | Pause and review |
| **Total API** | >$500 total | Switch to verifiable only |

### 12.2 Decision Tree

```
Week 3:
├── +5% improvement? → Continue
├── No improvement but system works? → Debug, adjust
└── System broken? → STOP
```

---

## 13. Ablation Plan

### 13.1 Required Ablations

| Ablation | What It Tests |
|----------|---------------|
| Baseline SFT | Same data, no competition |
| No frontier | Self-play only |
| Verifiable only | No judge-based |

### 13.2 Statistical Requirements

- 3 seeds minimum
- Report mean ± std
- p < 0.05 for significance

---

## 14. Risk Analysis

### 14.1 Top Risk: Reward Hacking

If success depends on a judge, models will game the judge.

**Mitigation:** Use verifiable tasks first.

### 14.2 Technical Risks

| Risk | Mitigation |
|------|------------|
| Training instability | Use DPO/GRPO (stable) |
| Judge gaming | Verifiable tasks first |
| Cost overrun | Hard budget gates |

---

## 15. Success Metrics

### 15.1 Phase 1 Success

- [ ] Verifier harness working
- [ ] +5% on domain benchmark
- [ ] Training loop runs 24h without crash

### 15.2 Ultimate Success

- [ ] Specialist beats frontier on target domain
- [ ] Weekly refresh cycle operational
- [ ] Paying customers

---

## Appendix: Minimal Blueprint

**Fastest path to value:**

1. **Pick one product:** Detection Engineer OR Secure Code Review
2. **Build verifier harness first** (this is the moat)
3. Generate initial dataset via frontier + verification
4. Train first specialist with DPO
5. Put behind router with frontier fallback
6. **Every week:** Mine failures → regenerate → retrain → ship

---

*Version 2.0 - Updated with Continuous Specialization Flywheel architecture*
