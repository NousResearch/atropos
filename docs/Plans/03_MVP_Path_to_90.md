# HYDRA Execution Plan: From 84 to 90+

**Purpose:** Close the gap between document quality (92) and execution viability (76)
**Core insight:** The plan is good enough. Now we need proof points, not more writing.
**Training Platform:** Atropos (this repository). Harness → BaseEnv → GRPO trainer.

---

## The Real Gaps (Why 84, Not 95)

| Gap | Type | Impact | Fix |
|-----|------|--------|-----|
| Benchmark doesn't run | Execution | Blocks all iteration | Build harness first, tasks second |
| 30 tasks is ~8 weeks work | Timeline | Delays everything | MVP with 10 tasks |
| Verifier hardening unproven | Risk | Sandbox escapes = game over | Escape test suite + drill |
| Go-to-market unclear | Strategy | Great tech, no customers | Define first buyer NOW |
| Ops overhead fear | Culture | Security slows refresh | Tiered controls by risk |

---

## 1. Minimum Viable Benchmark (10 tasks, not 30)

**Problem:** 30 tasks × (vuln repo + exploit + tests + goldens + 5 primary + 20 hidden) = 6-10 weeks

**Solution:** Prove the loop with 10 tasks first. Expand only after loop works.

### SecureCodeReview-10 (MVP)

```yaml
mvp_benchmark:
  name: "SecureCodeReview-10"
  purpose: "Prove the flywheel works before building full suite"

  tasks:
    sql_injection:
      - sqli-001: "String concatenation in Flask/Python"
      - sqli-002: "Parameterized query with type coercion (Node/Express)"

    xss:
      - xss-001: "dangerouslySetInnerHTML in React"
      - xss-002: "Template injection in Jinja2"

    path_traversal:
      - path-001: "User-controlled file path in Python"

    auth_bypass:
      - auth-001: "JWT none algorithm"
      - auth-002: "Timing attack in password comparison"

    ssrf:
      - ssrf-001: "User-controlled URL in requests.get()"

    logic:
      - logic-001: "Race condition in balance check"
      - logic-002: "Integer overflow in quantity calculation"

  variant_counts:
    primary_per_task: 3  # Not 5
    hidden_per_task: 10  # Not 20

  total_effort:
    tasks: 10
    primary_variants: 30
    hidden_variants: 100
    estimated_weeks: 3  # Not 8

  success_criteria:
    primary_pass_rate: ">= 80%"  # Lower bar for MVP
    hidden_pass_rate: ">= 70%"
    per_category_minimum: ">= 1 task per category"

  expansion_trigger:
    condition: "MVP loop running for 2+ weeks with stable metrics"
    action: "Expand to SecureCodeReview-30"
```

### Task Template (Speed Up Creation)

```yaml
# templates/sqli_template.yaml
# Reusable template for SQL injection tasks

template:
  category: sql_injection
  cwe: CWE-89

  vulnerable_code_pattern: |
    # Parameterize: {framework}, {language}, {injection_point}
    def {function_name}({params}):
        query = f"SELECT * FROM {table} WHERE {column} = '{user_input}'"
        return db.execute(query)

  exploit_pattern: |
    # Standard SQLi payload
    payload = "' OR '1'='1"
    # Framework-specific request

  test_pattern: |
    def test_{function_name}_normal():
        result = {function_name}("valid_input")
        assert result is not None

    def test_{function_name}_sql_chars():
        # Should not crash with SQL chars in input
        result = {function_name}("O'Brien")
        assert result is not None

  golden_pattern:
    inputs:
      - "valid_input"
      - "O'Brien"
      - ""
    # Outputs recorded during task creation

  variants:
    primary:
      - "Different table/column names"
      - "Different user input source"
      - "Slightly different query structure"
    hidden:
      - "Obfuscated payload"
      - "Unicode variations"
      - "Multi-statement injection"
```

### Build Order (Dependencies)

```
Week 1:
├── Day 1-2: Harness infrastructure
│   ├── runner.py skeleton
│   ├── Dockerfile with pinned deps
│   └── CI workflow (run_primary.sh)
│
├── Day 3-4: First task end-to-end
│   ├── sqli-001 complete (vuln + exploit + tests + goldens)
│   └── Verify: harness runs, produces JSON
│
└── Day 5: Hidden set infrastructure
    ├── git-crypt setup
    ├── run_hidden.sh (CI-only)
    └── Verify: fails locally, works in CI

Week 2:
├── Day 1-3: Tasks 2-5 using templates
├── Day 4-5: Tasks 6-10
└── Baselines: Run frontier model, record results

Week 2.5: Atropos + RelayOne Integration
├── Day 1: Wrap harness in SecureCodeReviewEnv (BaseEnv subclass)
├── Day 2: Test collect_trajectories produces valid ScoredDataGroup
├── Day 2: Start RelayOne devmode (docker-compose.devmode.yml)
│   └── Verify: POST /gateway/demo/agents/:id/invoke returns 200
└── Verify: python secure_code_review_env.py serve → connects to API

Week 3: First Specialist Training (on Atropos + RelayOne)
├── Day 1-2: Run training
│   ├── Terminal 0: cd mvp/deploy/docker && docker-compose -f docker-compose.devmode.yml up
│   ├── Terminal 1: python -m atroposlib.cli.run_api --port 8000
│   ├── Terminal 2: vllm serve Qwen/Qwen2.5-7B-Instruct --port 9004
│   ├── Terminal 3: python environments/hydra/secure_code_review_env.py serve
│   └── Terminal 4: python example_trainer/grpo.py --api-url http://localhost:8000
├── Day 3: Evaluate on MVP benchmark
├── Day 4-5: First refresh cycle (with DRQ attacker)
├── Milestone: Loop proven on Atropos
└── Milestone: Audit logs exportable from RelayOne (POST /api/v1/audit/export)
```

---

## 2. Harness-First Development

**Key insight:** The harness is more important than the tasks. A working harness with 3 tasks beats a spec for 30.

### Minimum Viable Harness

```python
# harness/runner.py - MVP version

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    exploit_before: bool
    exploit_after: bool
    tests_passed: bool
    scanner_clean: bool
    failure_reason: str | None

class BenchmarkRunner:
    """Minimum viable benchmark runner."""

    def __init__(self, tasks_dir: str, sandbox_image: str):
        self.tasks_dir = Path(tasks_dir)
        self.sandbox_image = sandbox_image

    def run_task(self, task_id: str, patched_code: str) -> TaskResult:
        """Run a single task and return results."""
        task_dir = self.tasks_dir / task_id

        # Load task config
        config = self._load_task_config(task_dir)

        # Step 1: Run exploit on original (should succeed)
        exploit_before = self._run_exploit(
            code=self._read_file(task_dir / "vulnerable" / config["vulnerable_file"]),
            exploit=task_dir / "exploit" / config["exploit_script"]
        )

        # Step 2: Run exploit on patched (should fail)
        exploit_after = self._run_exploit(
            code=patched_code,
            exploit=task_dir / "exploit" / config["exploit_script"]
        )

        # Step 3: Run regression tests
        tests_passed = self._run_tests(
            code=patched_code,
            test_dir=task_dir / "tests"
        )

        # Step 4: Run scanner
        scanner_clean = self._run_scanner(patched_code)

        # Determine pass/fail
        passed = (
            exploit_before and  # Original was vulnerable
            not exploit_after and  # Patched is not vulnerable
            tests_passed and  # Didn't break functionality
            scanner_clean  # No new issues introduced
        )

        failure_reason = None
        if not passed:
            if not exploit_before:
                failure_reason = "original_not_vulnerable"
            elif exploit_after:
                failure_reason = "patch_ineffective"
            elif not tests_passed:
                failure_reason = "regression_test_failure"
            elif not scanner_clean:
                failure_reason = "scanner_findings"

        return TaskResult(
            task_id=task_id,
            passed=passed,
            exploit_before=exploit_before,
            exploit_after=exploit_after,
            tests_passed=tests_passed,
            scanner_clean=scanner_clean,
            failure_reason=failure_reason
        )

    def _run_exploit(self, code: str, exploit: Path) -> bool:
        """Run exploit in sandbox, return True if exploit succeeded."""
        result = subprocess.run(
            ["docker", "run", "--rm", "--network=none",
             "-v", f"{exploit}:/exploit.py:ro",
             self.sandbox_image,
             "python", "/exploit.py"],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0

    def _run_tests(self, code: str, test_dir: Path) -> bool:
        """Run tests in sandbox, return True if all pass."""
        result = subprocess.run(
            ["docker", "run", "--rm", "--network=none",
             "-v", f"{test_dir}:/tests:ro",
             self.sandbox_image,
             "pytest", "/tests", "-v"],
            capture_output=True,
            timeout=60
        )
        return result.returncode == 0

    def _run_scanner(self, code: str) -> bool:
        """Run Semgrep, return True if no findings."""
        # Write code to temp file, run semgrep
        result = subprocess.run(
            ["semgrep", "--config=auto", "--json", "-"],
            input=code.encode(),
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            return False
        findings = json.loads(result.stdout)
        return len(findings.get("results", [])) == 0


def main():
    """Entry point for run_primary.sh"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="tasks")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.tasks_dir, "hydra-sandbox:latest")

    # Load task manifest
    manifest = json.load(open(f"{args.tasks_dir}/task_manifest.json"))

    results = []
    for task_id in manifest["tasks"]:
        # Get model's patch for this task
        patched = get_model_patch(args.model, task_id)
        result = runner.run_task(task_id, patched)
        results.append(result)

    # Write results
    summary = {
        "model": args.model,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "pass_rate": sum(1 for r in results if r.passed) / len(results),
        "results": [vars(r) for r in results]
    }
    json.dump(summary, open(args.output, "w"), indent=2)
    print(f"Results: {summary['passed']}/{summary['total']} passed")


if __name__ == "__main__":
    main()
```

### run_primary.sh (MVP)

```bash
#!/bin/bash
set -euo pipefail

# Build sandbox image if needed
if ! docker image inspect hydra-sandbox:latest &>/dev/null; then
    docker build -t hydra-sandbox:latest harness/
fi

# Run benchmark
python harness/runner.py \
    --tasks-dir tasks \
    --model "${MODEL:-claude-opus-4.5}" \
    --output results/primary_$(date +%Y%m%d_%H%M%S).json

echo "Primary benchmark complete"
```

---

## 3. Go-to-Market Clarity

**Problem:** Great technical plan, unclear who buys first and why.

### First Buyer Profile

```yaml
first_buyer:
  who: "Security-conscious startup/scale-up (50-500 engineers)"

  why_them:
    - Small security team (1-3 people), overwhelmed by SAST findings
    - Already using Semgrep/Snyk, drowning in alerts
    - Can't afford dedicated AppSec for every PR
    - Fast decision-making (no 6-month procurement)
    - Willing to try new tools if measurable ROI

  not_first_buyer:
    - Enterprise (too slow, too many requirements)
    - Tiny startup (no security budget, no process)
    - Regulated industries (need SOC2/HIPAA first)

  pain_point:
    "We get 50+ SAST findings per week. Engineers ignore them because
    too many false positives and no clear fix guidance."

  value_prop:
    "HYDRA takes your Semgrep findings and generates verified patches.
    You review the patch, not the finding. 50% fewer FPs, 3x faster fixes."
```

### First Workflow Wedge

```yaml
first_wedge:
  name: "Semgrep Finding → Verified Patch"

  why_this_wedge:
    - Semgrep has good adoption, generates machine-readable findings
    - Findings include file path + line + rule ID
    - Model can generate targeted patch
    - Verification is clear: does exploit still work?
    - ROI is measurable: time from finding to merged fix

  not_first_wedge:
    - "General security review" (too broad, hard to verify)
    - "Write detection rules" (different buyer, different toolchain)
    - "CTF assistance" (liability, governance overhead)

  integration:
    input: "Semgrep SARIF output"
    output: "PR with patch + evidence bundle"
    workflow:
      1. Semgrep runs in CI, outputs SARIF
      2. HYDRA ingests SARIF findings
      3. For each finding:
         a. Generate patch candidate
         b. Verify: exploit fails, tests pass, scanner clean
         c. If verified: create PR with evidence
         d. If not verified: flag for human review
      4. Human reviews PR, merges or requests changes
      5. Feedback logged for training

  metrics:
    - "Time from Semgrep finding to merged fix"
    - "False positive rate (verified patch rejected by human)"
    - "Automation rate (% of findings with auto-generated patch)"

  mvp_scope:
    languages: [python]  # Start with one
    finding_types: [sql_injection, xss, path_traversal]  # Match benchmark
    integration: "GitHub App only"  # Not GitLab yet
```

### Competitive Positioning (Sharper)

```yaml
positioning:
  vs_semgrep_autofix:
    them: "Generates fixes for simple patterns"
    us: "Verifies the fix actually works against an exploit"
    wedge: "Your autofix might introduce a regression. Ours is tested."

  vs_copilot_security:
    them: "Suggests security improvements inline"
    us: "Takes your existing findings, generates verified patches"
    wedge: "Copilot doesn't know your SAST rules. We integrate with them."

  vs_snyk_fix:
    them: "Fixes dependency vulnerabilities"
    us: "Fixes code vulnerabilities"
    wedge: "Snyk does deps. We do the code that uses those deps."

  headline:
    "From finding to verified fix in one PR"

  not_our_claim:
    - "Beat frontier models" (invites wrong comparison)
    - "AI security expert" (too broad)
    - "Replace your security team" (threatening, unrealistic)
```

---

## 4. Operational Simplification

**Problem:** Security processes could slow refresh to monthly (kills flywheel).

### Tiered Controls by Risk

```yaml
tiered_operations:
  principle: "Match controls to risk, not one-size-fits-all"

  tier_1_low_risk:
    applies_to:
      - Public code analysis
      - Open-source benchmark improvements
      - Internal tooling changes
    controls:
      - Standard code review
      - CI checks pass
      - No special approval
    refresh_impact: "None - can ship daily"

  tier_2_medium_risk:
    applies_to:
      - Model weight updates
      - Training data changes
      - Benchmark task additions
    controls:
      - Contamination check
      - One reviewer (ML or Security)
      - Automated quality gates
    refresh_impact: "Minimal - same-day approval"

  tier_3_high_risk:
    applies_to:
      - Hidden set changes
      - Customer data handling changes
      - Security policy changes
    controls:
      - Security team review
      - Two-person approval
      - Audit logging
    refresh_impact: "1-2 day approval cycle"

  tier_4_critical:
    applies_to:
      - Production incident response
      - Customer data access
      - Capability risk changes
    controls:
      - CTO + Security lead approval
      - Full audit trail
      - Post-action review
    refresh_impact: "N/A - not part of refresh"

  weekly_refresh_path:
    what_changes: "Model weights + training data"
    tier: 2
    controls: "Contamination check + one reviewer"
    bottleneck: "Reviewer availability"
    mitigation: "Async review, 4-hour SLA"
```

### Automation vs. Manual

```yaml
automation_targets:
  automate:
    - Contamination checks (every training run)
    - Benchmark execution (every PR)
    - Freeze trigger detection (continuous)
    - Access logging (always on)
    - Credential rotation (scheduled)

  semi_automate:
    - Model deployment (CI + one approval)
    - Training data addition (check + review)
    - Benchmark expansion (template + review)

  keep_manual:
    - Hidden set access (break-glass)
    - Customer data access (ticket-based)
    - Freeze decisions (human judgment)
    - Policy changes (deliberate)

  overhead_budget:
    target: "< 2 hours/week on security ops"
    breakdown:
      - Reviewer duties: 1 hour
      - Monitoring review: 30 min
      - Access request handling: 30 min
    excess_action: "Automate or eliminate"
```

---

## 5. Proof Point Checklist

**What actually raises the score from 84 to 90+:**

```yaml
proof_points:
  benchmark_proof:
    description: "SecureCodeReview-10 runs end-to-end"
    evidence:
      - [ ] run_primary.sh produces valid JSON
      - [ ] At least 10 tasks implemented
      - [ ] Hidden set encrypted, CI-only verified
      - [ ] Baseline table: 1 frontier + 1 tool
    owner: "ML Lead"
    target_date: "Week 3"
    score_impact: "+3 points"

  security_proof:
    description: "SSL3 audit on internal workflow"
    evidence:
      - [ ] Access controls verified (role separation)
      - [ ] Contamination check running
      - [ ] No hidden set in training logs
      - [ ] Frontier API logging operational
    owner: "Security Lead"
    target_date: "Week 4"
    score_impact: "+2 points"

  ops_proof:
    description: "Freeze drill completed"
    evidence:
      - [ ] Simulated trigger (inject metric anomaly)
      - [ ] Freeze invoked within 4 hours
      - [ ] Rollback executed successfully
      - [ ] Post-mortem documented
    owner: "Platform Lead"
    target_date: "Week 5"
    score_impact: "+2 points"

  customer_proof:
    description: "One pilot with measured ROI"
    evidence:
      - [ ] Pilot agreement signed
      - [ ] Integration deployed (GitHub App)
      - [ ] Baseline metrics collected (pre-HYDRA)
      - [ ] 2-week pilot completed
      - [ ] ROI metric reported (time to fix, FP rate)
    owner: "Product Lead"
    target_date: "Week 8"
    score_impact: "+3 points"

  total_potential: "+10 points → 94/100"
```

### Dependency Graph

```
Week 1-2: Harness + First Tasks
    │
    ▼
Week 3: Benchmark Proof ────────────────┐
    │                                    │
    ▼                                    │
Week 4: Security Proof ─────────────────┤
    │                                    │
    ▼                                    │
Week 5: Ops Proof (Freeze Drill) ───────┤
    │                                    │
    ▼                                    │
Week 6-7: First Specialist + Refresh ───┤
    │                                    │
    ▼                                    ▼
Week 8: Customer Pilot ──────────────► 90+ Score
```

---

## 6. Realistic Timeline (Revised)

**Old timeline:** 10 weeks to Phase 3 (optimistic)
**Realistic timeline:** 12-14 weeks with buffer

```yaml
timeline:
  phase_0_extended:
    duration: "3 weeks (was 2)"
    reason: "Harness infrastructure is always slower than expected"
    deliverables:
      week_1:
        - Harness skeleton (runner.py)
        - Dockerfile with pinned deps
        - CI workflow (run_primary.sh)
        - First task (sqli-001) end-to-end
      week_2:
        - Tasks 2-5 using templates
        - Hidden set infrastructure
        - run_hidden.sh CI-only
      week_3:
        - Tasks 6-10
        - Baseline collection (frontier + tool)
        - Benchmark proof checkpoint

  phase_1_realistic:
    duration: "4 weeks (was 3)"
    reason: "First training run always has surprises"
    deliverables:
      week_4:
        - First specialist training
        - Evaluate on MVP benchmark
        - SSL3 audit (security proof)
      week_5:
        - Freeze drill (ops proof)
        - First refresh cycle
        - Evidence bundle implementation
      week_6:
        - GitHub App MVP
        - API endpoint
        - SARIF integration
      week_7:
        - Internal dogfooding
        - Bug fixes
        - Documentation

  phase_2_realistic:
    duration: "4 weeks (was 3)"
    deliverables:
      week_8:
        - Pilot customer onboarding
        - Baseline metrics collection
      week_9-10:
        - Pilot execution
        - Feedback collection
      week_11:
        - ROI measurement
        - Customer proof checkpoint
        - Expand to 30 tasks (if loop works)

  phase_3:
    duration: "ongoing"
    start_condition: "After customer proof"
    deliverables:
      - Second vertical (Detection Engineer)
      - Scale customers
      - Weekly refresh operational

  total: "11 weeks to customer proof (was 6)"
  buffer: "3 weeks (for unknowns)"
  realistic_total: "14 weeks"
```

---

## 7. Risk Mitigations for Top Failure Modes

```yaml
failure_mode_mitigations:
  harness_drag:
    risk: "Benchmark takes too long, iteration stalls"
    mitigations:
      - MVP with 10 tasks, not 30
      - Templates for task creation
      - Harness-first development (working harness > more tasks)
      - Weekly milestone reviews (catch slippage early)
    owner: "ML Lead"

  false_confidence:
    risk: "Hidden set leaks, results look great until real use"
    mitigations:
      - Contamination check is automated, not manual
      - Hash-based detection, not just path-based
      - Quarterly hidden set rotation
      - Customer feedback as ground truth (not just benchmarks)
    owner: "Security Lead"

  verifier_gaming:
    risk: "Passing but bad patches"
    mitigations:
      - Redundant verifiers (scanner + dynamic)
      - Behavior preservation checks with goldens
      - Human spot-check on 10% of outputs
      - Customer rejection as training signal
    owner: "ML Lead"

  ops_overhead:
    risk: "Security processes slow refresh to monthly"
    mitigations:
      - Tiered controls (not everything is critical)
      - Automation budget (< 2 hours/week manual)
      - Async review with SLA (not blocking)
      - Quarterly overhead review
    owner: "Platform Lead"

  enterprise_friction:
    risk: "Customers refuse frontier fallback"
    mitigations:
      - Opt-out is built from day 1
      - Specialist-only mode available
      - Transparency on when frontier is used
      - VPC deployment on roadmap (not MVP)
    owner: "Product Lead"
```

---

## 8. Decision Log Template

**For tracking decisions that affect the plan:**

```yaml
decision_log:
  - id: "D001"
    date: "2026-01-15"
    decision: "MVP with 10 tasks, not 30"
    rationale: "Prove loop works before scaling"
    alternatives_considered:
      - "Full 30 tasks" (rejected: too slow)
      - "5 tasks" (rejected: insufficient coverage)
    owner: "ML Lead"
    status: "approved"

  - id: "D002"
    date: "2026-01-15"
    decision: "First wedge is Semgrep → Verified Patch"
    rationale: "Clear input, measurable output, existing tooling"
    alternatives_considered:
      - "General PR review" (rejected: too broad)
      - "Detection rules" (rejected: different buyer)
    owner: "Product Lead"
    status: "approved"

  - id: "D003"
    date: "2026-01-15"
    decision: "Python only for MVP"
    rationale: "Match benchmark, limit scope"
    alternatives_considered:
      - "Python + JavaScript" (rejected: doubles scope)
    owner: "ML Lead"
    status: "approved"
```

---

## Summary: Path to 90+

| Current | Gap | Fix | Target |
|---------|-----|-----|--------|
| 84/100 | Benchmark doesn't run | Build harness, 10 tasks | +3 |
| - | Security unproven | SSL3 audit | +2 |
| - | Ops unproven | Freeze drill | +2 |
| - | No customer | Pilot with ROI | +3 |
| **Target** | - | - | **94/100** |

**The path is clear. Stop writing, start building.**

Week 1 deliverable: `run_primary.sh` runs with 1 task and produces JSON.

Everything else follows from that.

---

*Document Version: 1.0*
*Status: Execution plan (not more specification)*
