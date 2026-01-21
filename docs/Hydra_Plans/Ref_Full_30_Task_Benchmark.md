# SecureCodeReview-30 Benchmark Specification

**Status:** Specification (not yet implemented)
**Version:** 1.0
**Purpose:** Turn "domain benchmark" into a runnable CI artifact

---

## 1. Repository Structure

```
benchmarks/
├── secure-code-review-30/
│   ├── README.md
│   ├── run_primary.sh          # Runs primary test suite
│   ├── run_hidden.sh           # CI-only, requires SCR30_HIDDEN_KEY
│   ├── baseline_results.json   # Frontier + tool baselines
│   │
│   ├── harness/
│   │   ├── Dockerfile          # Pinned environment
│   │   ├── requirements.txt    # Pinned Python deps
│   │   ├── package-lock.json   # Pinned Node deps
│   │   ├── runner.py           # Test orchestrator
│   │   ├── verifiers/
│   │   │   ├── exploit.py      # Sandbox exploit verification
│   │   │   ├── coverage.py     # Coverage measurement
│   │   │   ├── behavior.py     # Behavior preservation checks
│   │   │   ├── mutation.py     # Mutation testing wrapper
│   │   │   └── static.py       # Semgrep/scanner integration
│   │   └── anti_gaming.py      # Composite anti-gaming checks
│   │
│   ├── tasks/
│   │   ├── sqli-001/
│   │   │   ├── task.yaml       # Task definition
│   │   │   ├── vulnerable/     # Input code
│   │   │   ├── exploit/        # Working exploit
│   │   │   ├── tests/          # Regression tests
│   │   │   ├── golden/         # Expected behavior snapshots
│   │   │   └── primary_variants/  # 5 visible test variants
│   │   ├── sqli-002/
│   │   ├── ... (28 more tasks)
│   │   └── task_manifest.yaml  # All task IDs and categories
│   │
│   ├── hidden/                 # ENCRYPTED, CI-only access
│   │   ├── .gitattributes      # git-crypt configuration
│   │   ├── variants/           # Hidden test variants per task
│   │   └── corpus/             # Additional edge cases
│   │
│   └── results/
│       ├── schema.json         # Output format spec
│       └── .gitkeep
```

---

## 2. Task Definition Schema

Each task is fully specified in `task.yaml`:

```yaml
# tasks/sqli-001/task.yaml
id: sqli-001
version: 1
category: sql_injection
subcategory: string_concatenation

metadata:
  language: python
  framework: flask
  difficulty: medium
  cwe: CWE-89
  created: 2026-01-10
  author: security-team

input:
  vulnerable_file: vulnerable/app.py
  entry_point: "handle_search"
  context_files:
    - vulnerable/models.py
    - vulnerable/db.py
  vulnerability_description: |
    String concatenation in SQL query allows injection.
    User input from `request.args.get('q')` flows directly
    into f-string SQL query without parameterization.

exploit:
  script: exploit/poc.py
  expected_result: "data_exfiltration"
  timeout_seconds: 30

regression_tests:
  directory: tests/
  runner: pytest
  minimum_tests: 5
  required_coverage: 0.80

behavior_preservation:
  type: golden_responses
  golden_directory: golden/
  comparison_method: json_equivalence
  tolerance: exact  # or: semantic, numeric_epsilon

success_criteria:
  # ALL must pass for task success
  exploit_fails: true
  regression_tests_pass: true
  coverage_minimum: 0.80
  mutation_score_minimum: 0.60
  scanner_clean: true
  behavior_preserved: true
  no_functionality_removed: true

primary_variants:
  count: 5
  directory: primary_variants/
  description: "Visible during development, not training"

hidden_variants:
  count: 20
  directory: hidden/variants/sqli-001/
  access: ci_only
```

---

## 3. Success Criteria (Unified)

**Resolving the threshold inconsistency:**

```yaml
# Benchmark-level success definition
success_definition:
  # Pattern: Aggregate thresholds + category minimums

  aggregate:
    primary_pass_rate: ">= 90%"    # 27/30 tasks on primary variants
    hidden_pass_rate: ">= 85%"    # On hidden variants

  per_category_minimums:
    # No category can be completely failed
    sql_injection: ">= 4/5 tasks"
    xss: ">= 4/5 tasks"
    path_traversal: ">= 2/3 tasks"
    auth_bypass: ">= 3/4 tasks"
    ssrf: ">= 2/3 tasks"
    deserialization: ">= 2/3 tasks"
    dependency_vulns: ">= 3/4 tasks"
    logic_bugs: ">= 2/3 tasks"

  task_pass_definition:
    # A single task passes if ALL criteria met:
    - exploit_fails
    - regression_tests_pass
    - coverage >= 0.80
    - mutation_score >= 0.60
    - scanner_clean
    - behavior_preserved

  # No per-task percentage threshold (was confusing)
  # Task either passes (all criteria) or fails
```

---

## 4. Behavior Preservation (Concrete Definitions)

**The hard part made testable:**

```python
# harness/verifiers/behavior.py

class BehaviorPreservation:
    """Concrete behavior preservation checks per task type."""

    async def verify(self, task: Task, original: str, patched: str) -> BehaviorResult:
        method = task.behavior_preservation.type

        if method == "golden_responses":
            return await self.golden_response_check(task, patched)
        elif method == "api_contract":
            return await self.api_contract_check(task, original, patched)
        elif method == "cli_output":
            return await self.cli_output_check(task, original, patched)
        elif method == "property_invariants":
            return await self.property_check(task, patched)
        else:
            raise ValueError(f"Unknown behavior method: {method}")

    async def golden_response_check(self, task: Task, patched: str) -> BehaviorResult:
        """
        Compare outputs against pre-recorded golden responses.

        Golden responses are:
        - Pre-recorded during task creation
        - Cover all non-malicious input cases
        - Stored as JSON/text snapshots
        """
        golden_dir = task.behavior_preservation.golden_directory
        golden_files = list(Path(golden_dir).glob("*.json"))

        failures = []
        for golden_file in golden_files:
            expected = json.load(golden_file.open())
            input_data = expected["input"]
            expected_output = expected["output"]

            # Run patched code with same input
            actual_output = await self.sandbox.run(
                patched,
                entry_point=task.input.entry_point,
                input_data=input_data
            )

            # Compare based on tolerance
            if task.behavior_preservation.tolerance == "exact":
                if actual_output != expected_output:
                    failures.append({
                        "golden": golden_file.name,
                        "expected": expected_output,
                        "actual": actual_output
                    })
            elif task.behavior_preservation.tolerance == "semantic":
                if not self.semantic_equivalent(expected_output, actual_output):
                    failures.append(...)

        return BehaviorResult(
            passed=len(failures) == 0,
            failures=failures,
            total_checks=len(golden_files)
        )

    async def api_contract_check(self, task: Task, original: str, patched: str) -> BehaviorResult:
        """
        Verify API contract preserved using OpenAPI/schema comparison.

        Used for: REST endpoints, library functions with typed signatures.
        """
        original_schema = await self.extract_api_schema(original)
        patched_schema = await self.extract_api_schema(patched)

        # Check: all original endpoints still exist
        # Check: response schemas unchanged
        # Check: required parameters unchanged
        breaking_changes = self.detect_breaking_changes(original_schema, patched_schema)

        return BehaviorResult(
            passed=len(breaking_changes) == 0,
            failures=breaking_changes
        )

    async def property_check(self, task: Task, patched: str) -> BehaviorResult:
        """
        Verify invariants hold using Hypothesis-style property tests.

        Used for: Functions with clear mathematical properties,
        transformations that should be reversible, etc.
        """
        property_file = task.behavior_preservation.property_file

        # Run Hypothesis tests
        result = await self.sandbox.run_pytest(
            patched,
            test_file=property_file,
            hypothesis_settings={"max_examples": 100}
        )

        return BehaviorResult(
            passed=result.exit_code == 0,
            failures=result.failures if result.exit_code != 0 else []
        )
```

### Behavior Preservation Per Category

| Category | Primary Method | Fallback |
|----------|---------------|----------|
| SQL Injection | Golden responses (query results) | API contract |
| XSS | Golden responses (rendered HTML) | Property tests |
| Path Traversal | Golden responses (file contents) | CLI output |
| Auth Bypass | API contract (auth flows) | Golden responses |
| SSRF | Golden responses (request targets) | Property tests |
| Deserialization | Golden responses | Property invariants |
| Dependency Vulns | Regression tests only | N/A |
| Logic Bugs | Golden responses | Property tests |

---

## 5. Hidden Set Handling (Leak Prevention)

**This is the integrity of the entire benchmark.**

### Storage & Access Control

```yaml
hidden_set_policy:
  storage:
    location: benchmarks/secure-code-review-30/hidden/
    encryption: git-crypt with AES-256
    key_management:
      - Production key: CI/CD secrets only
      - No local decryption for developers
      - Key rotation: quarterly

  access_control:
    # Who can decrypt
    ci_systems:
      - github_actions: via SCR30_HIDDEN_KEY secret
      - internal_ci: via vault secret

    # Who cannot decrypt
    blocked:
      - developer_machines
      - training_pipelines
      - model_inference_systems

  audit:
    - All hidden set accesses logged
    - Alert on access from non-CI systems
    - Quarterly access review
```

### Execution Isolation

```bash
#!/bin/bash
# run_hidden.sh - CI-only execution

set -euo pipefail

# Verify CI environment
if [ -z "${CI:-}" ]; then
    echo "ERROR: run_hidden.sh can only run in CI"
    exit 1
fi

# Verify key is available
if [ -z "${SCR30_HIDDEN_KEY:-}" ]; then
    echo "ERROR: SCR30_HIDDEN_KEY not set"
    exit 1
fi

# Decrypt hidden set to tmpfs (never touches disk)
HIDDEN_DIR=$(mktemp -d -p /dev/shm)
trap "rm -rf $HIDDEN_DIR" EXIT

git-crypt unlock <(echo "$SCR30_HIDDEN_KEY" | base64 -d)
cp -r hidden/variants/* "$HIDDEN_DIR/"
git-crypt lock

# Run hidden evaluation
python harness/runner.py \
    --tasks tasks/task_manifest.yaml \
    --hidden-dir "$HIDDEN_DIR" \
    --output results/hidden_results.json

# Results contain pass/fail, never hidden test content
echo "Hidden evaluation complete"
```

### Training Pipeline Isolation

```python
# training/data_loader.py

class BenchmarkAwareDataLoader:
    """Ensures training never sees hidden test data."""

    FORBIDDEN_PATHS = [
        "benchmarks/*/hidden/",
        "benchmarks/*/primary_variants/",  # Also excluded from training
    ]

    def load_training_data(self, paths: List[str]) -> Dataset:
        for path in paths:
            for forbidden in self.FORBIDDEN_PATHS:
                if fnmatch.fnmatch(path, forbidden):
                    raise DataLeakageError(
                        f"Attempted to load benchmark data: {path}"
                    )

        # Also check file contents for benchmark task IDs
        # (defense against copy-paste leakage)
        return self._load_with_leak_detection(paths)

    def _load_with_leak_detection(self, paths: List[str]) -> Dataset:
        TASK_ID_PATTERN = re.compile(r"(sqli|xss|path|auth|ssrf|deser|dep|logic)-\d{3}")

        for path in paths:
            content = Path(path).read_text()
            if TASK_ID_PATTERN.search(content):
                raise DataLeakageError(
                    f"Training data contains benchmark task IDs: {path}"
                )

        return Dataset.from_files(paths)
```

---

## 6. Baseline Table Requirement

**No benchmark without baselines.**

```yaml
# baseline_results.json schema
baseline_results:
  benchmark_version: "1.0"
  last_updated: "2026-01-15"

  baselines:
    # Frontier model baselines
    - name: "claude-opus-4.5"
      type: frontier_model
      date: "2026-01-10"
      primary_pass_rate: 0.73  # 22/30 tasks
      hidden_pass_rate: 0.67   # On hidden variants
      per_category:
        sql_injection: 4/5
        xss: 4/5
        path_traversal: 2/3
        auth_bypass: 3/4
        ssrf: 2/3
        deserialization: 2/3
        dependency_vulns: 3/4
        logic_bugs: 2/3
      notes: "Struggles with behavior preservation on complex refactors"

    - name: "gpt-4.5"
      type: frontier_model
      date: "2026-01-10"
      primary_pass_rate: 0.70
      hidden_pass_rate: 0.63
      per_category: {...}

    # Tool-only baselines (what SAST + manual fix gets you)
    - name: "semgrep-autofix"
      type: static_tool
      date: "2026-01-10"
      primary_pass_rate: 0.40  # Many fixes don't preserve behavior
      hidden_pass_rate: 0.35
      notes: "Autofix works for simple patterns, fails on complex cases"

    - name: "codeql-suggestions"
      type: static_tool
      date: "2026-01-10"
      primary_pass_rate: 0.33
      hidden_pass_rate: 0.30

    # Your model versions
    - name: "hydra-v0.1"
      type: specialist
      date: "2026-01-20"
      primary_pass_rate: null  # To be filled
      hidden_pass_rate: null
      training_data_size: null
      notes: "Initial specialist, pre-optimization"

    - name: "hydra-v0.2"
      type: specialist
      date: null  # After first refresh
      primary_pass_rate: null
      hidden_pass_rate: null

  target:
    primary_pass_rate: ">= 0.90"
    hidden_pass_rate: ">= 0.85"
    vs_frontier: "+15% on hidden"  # The real claim
```

### Baseline Collection Script

```bash
#!/bin/bash
# collect_baselines.sh - Run all baselines

MODELS=(
    "claude-opus-4.5:anthropic"
    "gpt-4.5:openai"
)

TOOLS=(
    "semgrep-autofix"
    "codeql-suggestions"
)

for model in "${MODELS[@]}"; do
    name="${model%%:*}"
    provider="${model##*:}"

    echo "Running baseline: $name"
    python harness/runner.py \
        --model "$name" \
        --provider "$provider" \
        --tasks tasks/task_manifest.yaml \
        --output "results/baseline_${name}.json"
done

for tool in "${TOOLS[@]}"; do
    echo "Running tool baseline: $tool"
    python harness/runner.py \
        --tool "$tool" \
        --tasks tasks/task_manifest.yaml \
        --output "results/baseline_${tool}.json"
done

# Aggregate into baseline_results.json
python harness/aggregate_baselines.py results/baseline_*.json > baseline_results.json
```

---

## 7. Router Confidence Specification

**Making "confidence < threshold" concrete.**

```yaml
router_confidence:
  sources:
    # Primary: Self-consistency across multiple samples
    self_consistency:
      method: "sample_and_vote"
      samples: 5
      temperature: 0.7
      agreement_threshold: 0.8  # 4/5 must agree on approach
      weight: 0.4

    # Secondary: Verifier pre-checks (fast, cheap)
    verifier_precheck:
      checks:
        - syntax_valid
        - imports_resolve
        - no_obvious_deletions
      weight: 0.3

    # Tertiary: Task similarity to training distribution
    distribution_similarity:
      method: "embedding_distance"
      threshold: 0.85  # cosine similarity to nearest training example
      weight: 0.2

    # Quaternary: Model's own uncertainty (calibrated)
    self_reported:
      method: "probability_of_correct"
      calibration: "temperature_scaling"
      weight: 0.1

  composite_confidence:
    formula: "weighted_sum(sources)"
    fallback_threshold: 0.7

    decision:
      confidence >= 0.85: "specialist_only"
      0.7 <= confidence < 0.85: "specialist_with_verification"
      confidence < 0.7: "fallback_to_frontier"

  calibration:
    method: "temperature_scaling"
    calibration_set: "500 held-out examples"
    target_metric: "expected_calibration_error < 0.05"
    recalibration_frequency: "weekly with refresh"

    # Reliability diagram requirements
    reliability_check:
      bins: 10
      max_gap: 0.10  # Max difference between confidence and accuracy per bin

  monitoring:
    metrics:
      - name: "fallback_rate"
        target: "< 20%"
        alert_threshold: "> 30%"

      - name: "post_fallback_success"
        description: "Did frontier succeed when specialist was uncertain?"
        target: "> 80%"
        alert_threshold: "< 60%"  # Means fallback logic is broken

      - name: "expected_calibration_error"
        target: "< 0.05"
        alert_threshold: "> 0.10"

      - name: "silent_degradation_detection"
        method: "track accuracy drift over 7-day windows"
        alert_threshold: "> 5% drop"

    dashboards:
      - Confidence distribution over time
      - Fallback rate by task category
      - Reliability diagram (updated daily)
```

### Calibration Implementation

```python
# router/calibration.py

class ConfidenceCalibrator:
    """Temperature scaling for confidence calibration."""

    def __init__(self):
        self.temperature = 1.0  # Learned parameter

    def calibrate(self, calibration_set: List[Tuple[float, bool]]):
        """
        Learn temperature from held-out calibration set.

        Args:
            calibration_set: List of (raw_confidence, was_correct)
        """
        # Optimize temperature to minimize NLL
        def nll_loss(temp):
            total_loss = 0
            for raw_conf, correct in calibration_set:
                calibrated = self._apply_temperature(raw_conf, temp)
                if correct:
                    total_loss -= np.log(calibrated + 1e-10)
                else:
                    total_loss -= np.log(1 - calibrated + 1e-10)
            return total_loss / len(calibration_set)

        result = scipy.optimize.minimize_scalar(
            nll_loss,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        self.temperature = result.x

    def _apply_temperature(self, raw_confidence: float, temp: float) -> float:
        # Temperature scaling on logit
        logit = np.log(raw_confidence / (1 - raw_confidence + 1e-10))
        scaled_logit = logit / temp
        return 1 / (1 + np.exp(-scaled_logit))

    def expected_calibration_error(self, test_set: List[Tuple[float, bool]], bins: int = 10) -> float:
        """Compute ECE on test set."""
        bin_boundaries = np.linspace(0, 1, bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = [(c, correct) for c, correct in test_set if low <= c < high]

            if len(in_bin) > 0:
                accuracy = sum(correct for _, correct in in_bin) / len(in_bin)
                avg_confidence = sum(c for c, _ in in_bin) / len(in_bin)
                bin_accuracies.append(accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(len(in_bin))

        # Weighted average of |accuracy - confidence|
        total = sum(bin_counts)
        ece = sum(
            (count / total) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        )
        return ece
```

---

## 8. Compliance Status Labels

**Fixing the "too strong" claims.**

```yaml
compliance_status:
  current:
    - encryption_at_rest: "AES-256"
    - encryption_in_transit: "TLS 1.3"
    - access_logging: "All API calls logged"
    - data_retention_controls: "Configurable per customer"

  planned_q1_2026:
    - soc2_type_1: "Audit scheduled March 2026"
    - penetration_testing: "Annual, first test February 2026"
    - bug_bounty: "Launch after SOC2 Type 1"

  planned_q2_2026:
    - soc2_type_2: "Requires 6 months of Type 1"
    - iso_27001: "Certification process initiated"

  enterprise_roadmap:
    - hipaa_baa: "Available Q3 2026, requires VPC deployment"
    - fedramp_moderate: "2027 roadmap item, dependent on customer demand"
    - pci_dss: "Not planned unless customer demand"

  deployment_options:
    cloud_saas:
      status: current
      compliance: [encryption, logging, retention]
      data_residency: "US-only initially, EU Q2 2026"

    vpc_deployment:
      status: planned_q1_2026
      compliance: [all_cloud + customer_controlled_keys]
      data_residency: "Customer controlled"

    on_premises:
      status: planned_q2_2026
      compliance: [customer_responsible]
      licensing: "Annual enterprise license"
```

---

## 9. Output Format

**What run_primary.sh and run_hidden.sh produce:**

```json
{
  "benchmark": "SecureCodeReview-30",
  "version": "1.0",
  "run_type": "primary",
  "timestamp": "2026-01-15T14:30:00Z",
  "model": "hydra-v0.1",

  "summary": {
    "total_tasks": 30,
    "passed": 25,
    "failed": 5,
    "pass_rate": 0.833,
    "meets_threshold": false,
    "threshold": 0.90
  },

  "per_category": {
    "sql_injection": {"passed": 5, "total": 5, "meets_minimum": true},
    "xss": {"passed": 4, "total": 5, "meets_minimum": true},
    "path_traversal": {"passed": 2, "total": 3, "meets_minimum": true},
    "auth_bypass": {"passed": 3, "total": 4, "meets_minimum": true},
    "ssrf": {"passed": 3, "total": 3, "meets_minimum": true},
    "deserialization": {"passed": 2, "total": 3, "meets_minimum": true},
    "dependency_vulns": {"passed": 4, "total": 4, "meets_minimum": true},
    "logic_bugs": {"passed": 2, "total": 3, "meets_minimum": true}
  },

  "failures": [
    {
      "task_id": "xss-003",
      "failure_reason": "behavior_not_preserved",
      "details": "Golden response mismatch on input case 7"
    },
    {
      "task_id": "logic-002",
      "failure_reason": "mutation_score_low",
      "details": "Mutation score 0.45 < 0.60 threshold"
    }
  ],

  "anti_gaming_checks": {
    "coverage_failures": 0,
    "behavior_failures": 1,
    "mutation_failures": 1,
    "functionality_removed": 0
  }
}
```

---

## 10. Implementation Checklist

### Week 1: Harness Infrastructure
- [ ] Create repository structure
- [ ] Implement Dockerfile with pinned deps
- [ ] Implement runner.py skeleton
- [ ] Set up git-crypt for hidden directory
- [ ] Create CI workflow for run_primary.sh

### Week 2: First 10 Tasks
- [ ] sqli-001 through sqli-005 (SQL injection)
- [ ] xss-001 through xss-005 (XSS)
- [ ] Create golden responses for each
- [ ] Verify exploit scripts work
- [ ] Create 5 primary variants per task

### Week 3: Remaining Tasks + Baselines
- [ ] Complete all 30 tasks
- [ ] Run frontier model baselines
- [ ] Run tool baselines
- [ ] Populate baseline_results.json
- [ ] Create hidden variants (20 per task)

### Week 4: Hidden Set + Calibration
- [ ] Encrypt hidden directory
- [ ] Implement run_hidden.sh with CI-only checks
- [ ] Implement router confidence calibration
- [ ] Run first calibration on held-out set
- [ ] Verify ECE < 0.05

---

## Deliverable Definition

**The benchmark is "real" when:**

1. `./run_primary.sh` executes end-to-end and produces valid JSON
2. `./run_hidden.sh` fails locally but succeeds in CI
3. `baseline_results.json` contains at least 2 frontier models + 1 tool
4. All 30 tasks have working exploits, tests, and golden responses
5. Hidden variants exist and are encrypted
6. Training pipeline rejects any path matching `benchmarks/*/hidden/`

**Until these are true, the score stays at 70-78.**

---

*Specification Version: 1.0*
*This document describes what must be built, not what exists.*
