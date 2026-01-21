# HYDRA Security Operations Model

**Inspired by:** AI-2027 operational lessons
**Purpose:** Transform HYDRA from "product with security features" to "security-first product org"
**Version:** 1.0

---

## Executive Summary

AI-2027's core lessons apply directly to a security specialist product org:

| AI-2027 Lesson | HYDRA Translation |
|----------------|-------------------|
| "Never finishes learning" | Process is the product; ship many versions |
| "Looks good" optimization | Models game verifiers unless outputs carry proof |
| Security failure is default | Secrets theft >> weights theft; insider risk dominates |
| Racing dynamics | Speed pressure causes corners cut unless pre-committed gates exist |
| Concentration + weak oversight | Small teams + implicit governance = bad decisions at speed |

This document operationalizes these lessons for HYDRA.

---

## 1. Security Levels (WSL/SSL) as Phase Gates

### 1.1 Definitions

**Weights Security Level (WSL):** Preventing model artifact theft
- Checkpoints, LoRAs, merged weights
- Training infrastructure access
- Model registry

**Secrets Security Level (SSL):** Preventing high-value information leakage
- Customer code/logs
- Verifier bypass techniques
- Benchmark hidden sets
- Winning prompts / exploit reproductions
- Training data / failure mining outputs
- Proprietary evaluation results

### 1.2 Security Level Definitions

```yaml
security_levels:
  WSL:
    level_1:
      name: "Basic"
      description: "Development/prototype"
      controls:
        - Encrypted storage at rest
        - Basic access controls
        - No public exposure

    level_2:
      name: "Protected"
      description: "Internal production"
      controls:
        - All Level 1 controls
        - Audit logging on access
        - Separate training/inference environments
        - No weights on developer machines

    level_3:
      name: "Hardened"
      description: "Customer-facing production"
      controls:
        - All Level 2 controls
        - Hardware security modules for signing
        - Air-gapped training environment
        - Multi-party approval for weight export
        - Integrity verification on load

    level_4:
      name: "Maximum"
      description: "High-value / regulated"
      controls:
        - All Level 3 controls
        - Secure enclave execution
        - No network during inference (optional)
        - Full provenance chain

  SSL:
    level_1:
      name: "Basic"
      description: "Development"
      controls:
        - Encryption at rest
        - Role-based access
        - No customer data in dev environments

    level_2:
      name: "Protected"
      description: "Internal production"
      controls:
        - All Level 1 controls
        - Audit logging
        - Data classification labels
        - Retention policies enforced
        - No secrets in logs/errors

    level_3:
      name: "Restricted"
      description: "Customer-facing"
      controls:
        - All Level 2 controls
        - Least-privilege access (need-to-know)
        - Automated secret scanning
        - Customer data isolation (per-tenant)
        - No customer data to frontier APIs without opt-in
        - Hidden test sets CI-only

    level_4:
      name: "Compartmentalized"
      description: "Highly sensitive"
      controls:
        - All Level 3 controls
        - Separation of duties enforced
        - Two-person integrity for sensitive operations
        - Time-limited access grants
        - Full audit trail with tamper evidence
```

### 1.3 Phase Requirements

```yaml
phase_security_requirements:
  phase_0_verifier:
    WSL: 1  # Prototype, not deployed
    SSL: 2  # Never store customer secrets in training corpora
    hard_rules:
      - "No customer data in any corpus"
      - "Hidden sets exist but access not yet restricted"

  phase_1_mvp:
    WSL: 2  # Protected weights
    SSL: 2  # Protected secrets
    hard_rules:
      - "Model registry access-controlled"
      - "Training data provenance tracked"
      - "No hidden sets sent to frontier APIs"

  phase_2_router:
    WSL: 2
    SSL: 3  # Required before scaling customers
    hard_rules:
      - "Customer data isolation verified"
      - "Frontier fallback data handling documented"
      - "Hidden sets CI-only with audit"
    gate: "Security review before customer expansion"

  phase_3_scale:
    WSL: 3  # Hardened for production
    SSL: 3
    hard_rules:
      - "Two-person approval for model deploys"
      - "Incident response tested"

  enterprise:
    WSL: 3-4  # Customer dependent
    SSL: 4    # Compartmentalized
    hard_rules:
      - "Per-customer key management"
      - "Regulatory compliance verified"
```

### 1.4 Security Level Audit Checklist

```yaml
audit_checklist:
  ssl_3_verification:
    access_controls:
      - [ ] Customer data requires explicit grant
      - [ ] Grants are time-limited (max 30 days)
      - [ ] All accesses logged with reason
      - [ ] Quarterly access review conducted

    frontier_api_handling:
      - [ ] Customer data never sent without opt-in
      - [ ] Opt-in is per-request or per-session, not blanket
      - [ ] Data sent to frontier is logged
      - [ ] Customer can audit what was sent

    hidden_set_protection:
      - [ ] Hidden sets encrypted at rest
      - [ ] Decryption only in CI environment
      - [ ] No hidden set content in logs/errors
      - [ ] Hash-based contamination check in training pipeline

    separation:
      - [ ] Production log access != model push access
      - [ ] Training data access != customer data access
      - [ ] Hidden set access != training pipeline access
```

---

## 2. Benchmark Leakage Control System

### 2.1 The Threat Model

Benchmark integrity fails when:

| Leak Vector | How It Happens | Detection Difficulty |
|-------------|----------------|---------------------|
| Direct training | Hidden sets accidentally in training corpus | Medium (hash check) |
| Prompt leakage | Engineer pastes hidden task into prompt | Hard |
| Frontier exposure | Teacher model sees hidden test via API | Hard |
| Notebook leakage | Hidden examples in Jupyter notebooks | Medium |
| Slack/ticket leakage | Hidden task discussed in comms | Very hard |
| Gradual memorization | Model memorizes through repeated eval | Hard |

### 2.2 Mandatory Controls

```yaml
benchmark_integrity:
  dataset_management:
    immutable_ids:
      format: "{category}-{number}-{variant_hash}"
      example: "sqli-001-a3f2b1"
      rule: "Once assigned, ID never reused"

    signed_manifests:
      content:
        - task_id
        - sha256 of all task files
        - creation date
        - last_modification date
      signing: "GPG key held by security lead"
      verification: "CI checks signature before eval"

    version_control:
      primary_sets: "Git, normal access"
      hidden_sets: "git-crypt, CI-only decryption"
      manifest: "Git, signed commits only"

  contamination_detection:
    training_pipeline:
      - Hash all training examples
      - Compare against hidden set hashes before training
      - Block if overlap detected
      - Alert security team

    prompt_monitoring:
      - Log all prompts sent to frontier APIs
      - Scan for hidden task IDs (regex)
      - Scan for hidden task content (fuzzy match)
      - Alert on detection

    notebook_scanning:
      - Pre-commit hook scans .ipynb files
      - Blocks commits containing hidden set references
      - Weekly scan of all notebooks in org

  access_model:
    hidden_sets:
      who_can_access:
        - CI systems (automated)
        - Security lead (break-glass, logged)
      who_cannot_access:
        - ML engineers
        - Product engineers
        - Frontier API providers
      break_glass:
        - Requires security lead + CTO approval
        - Time-limited (4 hours max)
        - Full audit trail
        - Mandatory rotation of accessed subset

  frontier_api_rules:
    never_send:
      - Hidden test content
      - Hidden test IDs
      - Customer production data (without opt-in)
    always_log:
      - All prompts sent to frontier
      - Response content
      - Purpose/justification
```

### 2.3 Contamination Check Implementation

```python
# training/contamination_check.py

import hashlib
from pathlib import Path
from typing import Set

class ContaminationChecker:
    """Prevent hidden set leakage into training."""

    def __init__(self, hidden_manifest_path: str):
        self.hidden_hashes = self._load_hidden_hashes(hidden_manifest_path)
        self.hidden_task_ids = self._load_hidden_task_ids(hidden_manifest_path)

    def _load_hidden_hashes(self, manifest_path: str) -> Set[str]:
        """Load SHA256 hashes of all hidden set content."""
        # In practice: load from signed manifest
        # Manifest is readable; content is not
        manifest = json.load(open(manifest_path))
        return set(manifest["content_hashes"])

    def _load_hidden_task_ids(self, manifest_path: str) -> Set[str]:
        """Load all hidden task IDs for pattern matching."""
        manifest = json.load(open(manifest_path))
        return set(manifest["hidden_task_ids"])

    def check_training_batch(self, examples: List[str]) -> ContaminationResult:
        """Check a batch of training examples for contamination."""
        contaminated = []

        for i, example in enumerate(examples):
            # Hash-based check
            example_hash = hashlib.sha256(example.encode()).hexdigest()
            if example_hash in self.hidden_hashes:
                contaminated.append({
                    "index": i,
                    "reason": "exact_hash_match",
                    "severity": "critical"
                })
                continue

            # Fuzzy content check (n-gram overlap)
            if self._has_high_ngram_overlap(example):
                contaminated.append({
                    "index": i,
                    "reason": "high_ngram_overlap",
                    "severity": "warning"
                })
                continue

            # Task ID pattern check
            for task_id in self.hidden_task_ids:
                if task_id in example:
                    contaminated.append({
                        "index": i,
                        "reason": f"contains_hidden_task_id:{task_id}",
                        "severity": "critical"
                    })
                    break

        return ContaminationResult(
            clean=len(contaminated) == 0,
            contaminated_count=len(contaminated),
            details=contaminated
        )

    def check_prompt_for_frontier(self, prompt: str) -> bool:
        """Check if prompt is safe to send to frontier API."""
        # Never send hidden task IDs
        for task_id in self.hidden_task_ids:
            if task_id in prompt:
                raise ContaminationError(
                    f"Attempted to send hidden task ID to frontier: {task_id}"
                )

        # Check for hidden content patterns
        if self._has_high_ngram_overlap(prompt):
            raise ContaminationError(
                "Prompt has high overlap with hidden set content"
            )

        return True

    def _has_high_ngram_overlap(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text has suspiciously high n-gram overlap with hidden sets."""
        # Implementation: compare 5-grams against hidden set 5-grams
        # Return True if Jaccard similarity > threshold
        pass
```

---

## 3. Router as Risk-Aware Policy Engine

### 3.1 Routing Dimensions

The router is not just "confident → specialist, uncertain → frontier." It's a policy engine with multiple dimensions:

```yaml
routing_dimensions:
  confidence:
    description: "Model's calibrated certainty about correctness"
    source: "Self-consistency + verifier precheck + calibration"
    levels:
      high: ">= 0.85"
      medium: "0.7 - 0.85"
      low: "< 0.7"

  capability_risk:
    description: "Is this request asking for dangerous capabilities?"
    categories:
      low:
        - "Vulnerability analysis (read-only)"
        - "Detection rule generation"
        - "Security review (suggestions only)"
      medium:
        - "Patch generation"
        - "Exploit explanation"
        - "Attack chain documentation"
      high:
        - "Working exploit code"
        - "Lateral movement steps"
        - "Evasion techniques"
      blocked:
        - "Malware generation"
        - "Social engineering scripts"
        - "Real-world targeting"

  data_sensitivity:
    description: "Does the request contain sensitive data?"
    levels:
      public: "No customer data, no secrets"
      internal: "Internal code, non-production"
      customer: "Customer code/logs"
      regulated: "PII, PHI, financial data"

  actionability:
    description: "What can the model do with its output?"
    levels:
      analyze_only: "Read, no write"
      suggest: "Propose changes, human applies"
      apply_staged: "Apply to staging, human promotes"
      apply_production: "Apply directly (requires approval)"
      execute_sandboxed: "Run code in sandbox"
      execute_networked: "NEVER by default"
```

### 3.2 Routing Policy Matrix

```yaml
routing_policy:
  # Confidence routing
  confidence_routing:
    high_confidence:
      action: "specialist_only"
      verification: "post-hoc spot check (10%)"
    medium_confidence:
      action: "specialist_with_full_verification"
      verification: "all outputs verified before return"
    low_confidence:
      action: "fallback_to_frontier"
      verification: "frontier output also verified"
      logging: "log for future training"

  # Capability risk routing
  capability_risk_routing:
    low:
      allowed_models: [specialist, frontier]
      allowed_actions: [analyze_only, suggest, apply_staged]
    medium:
      allowed_models: [specialist, frontier]
      allowed_actions: [analyze_only, suggest]
      required: "audit_log"
    high:
      allowed_models: [specialist_only]  # No frontier exposure
      allowed_actions: [analyze_only]
      required: "human_review"
    blocked:
      action: "refuse"
      logging: "alert_security_team"

  # Data sensitivity routing
  data_sensitivity_routing:
    public:
      frontier_allowed: true
      logging: "standard"
    internal:
      frontier_allowed: true
      logging: "standard"
    customer:
      frontier_allowed: "opt_in_only"
      logging: "enhanced"
      retention: "customer_policy"
    regulated:
      frontier_allowed: false
      logging: "enhanced"
      retention: "regulatory_requirement"

  # Combined policy examples
  policy_examples:
    - scenario: "Customer asks for patch for their code"
      confidence: medium
      capability_risk: medium
      data_sensitivity: customer
      result:
        model: specialist
        action: suggest
        frontier: "only if customer opted in"
        verification: full
        logging: enhanced

    - scenario: "Internal security review of open source"
      confidence: high
      capability_risk: low
      data_sensitivity: public
      result:
        model: specialist
        action: apply_staged
        frontier: allowed
        verification: spot_check
        logging: standard

    - scenario: "Request for working exploit code"
      confidence: any
      capability_risk: high
      data_sensitivity: any
      result:
        model: specialist_only
        action: analyze_only
        frontier: never
        verification: full
        logging: alert
        human_review: required
```

### 3.3 Policy Engine Implementation

```python
# router/policy_engine.py

@dataclass
class RoutingDecision:
    model: str  # "specialist", "frontier", "refuse"
    allowed_actions: List[str]
    verification_level: str
    frontier_data_allowed: bool
    logging_level: str
    human_review_required: bool
    reason: str

class PolicyEngine:
    """Risk-aware routing policy engine."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.capability_classifier = CapabilityRiskClassifier()
        self.sensitivity_classifier = DataSensitivityClassifier()

    async def route(self, request: Request, confidence: float) -> RoutingDecision:
        # Classify request
        capability_risk = await self.capability_classifier.classify(request)
        data_sensitivity = await self.sensitivity_classifier.classify(request)

        # Check for blocked capabilities first
        if capability_risk == "blocked":
            return RoutingDecision(
                model="refuse",
                allowed_actions=[],
                verification_level="none",
                frontier_data_allowed=False,
                logging_level="alert",
                human_review_required=False,
                reason="blocked_capability"
            )

        # Determine model based on data sensitivity
        if data_sensitivity == "regulated":
            frontier_allowed = False
        elif data_sensitivity == "customer":
            frontier_allowed = request.customer_opted_in_frontier
        else:
            frontier_allowed = True

        # Determine model based on confidence
        if confidence >= 0.85:
            primary_model = "specialist"
        elif confidence >= 0.7:
            primary_model = "specialist"  # But with full verification
        else:
            primary_model = "frontier" if frontier_allowed else "specialist"

        # Override for high capability risk
        if capability_risk == "high":
            primary_model = "specialist"  # Never send to frontier
            frontier_allowed = False

        # Determine allowed actions
        allowed_actions = self._get_allowed_actions(capability_risk, data_sensitivity)

        # Determine verification level
        if confidence < 0.7 or capability_risk in ["medium", "high"]:
            verification_level = "full"
        elif confidence < 0.85:
            verification_level = "full"
        else:
            verification_level = "spot_check"

        # Determine human review requirement
        human_review = capability_risk == "high"

        return RoutingDecision(
            model=primary_model,
            allowed_actions=allowed_actions,
            verification_level=verification_level,
            frontier_data_allowed=frontier_allowed,
            logging_level=self._get_logging_level(data_sensitivity, capability_risk),
            human_review_required=human_review,
            reason=self._build_reason(confidence, capability_risk, data_sensitivity)
        )

    def _get_allowed_actions(self, capability_risk: str, data_sensitivity: str) -> List[str]:
        base_actions = {
            "low": ["analyze_only", "suggest", "apply_staged"],
            "medium": ["analyze_only", "suggest"],
            "high": ["analyze_only"]
        }
        return base_actions.get(capability_risk, ["analyze_only"])
```

---

## 4. Slowdown / Freeze Protocol

### 4.1 Pre-Committed Triggers

```yaml
freeze_protocol:
  description: |
    Pre-committed conditions that halt releases. The freeze must be:
    - Cheap to invoke
    - Socially acceptable ("any engineer can trigger a review")
    - Fast to resolve or escalate

  automatic_triggers:
    # Verifier gaming signals
    - name: "verifier_pass_spike"
      condition: "Verifier pass rate increases >10% week-over-week"
      paired_with: "No corresponding improvement on hidden set"
      action: freeze
      rationale: "Classic reward hacking signature"

    - name: "hidden_set_regression"
      condition: "Hidden set performance drops >5% from baseline"
      action: freeze
      rationale: "Model may be overfitting to primary set"

    - name: "verbosity_correlation"
      condition: "Success rate correlates >0.5 with output length"
      action: freeze
      rationale: "Judge/verifier gaming via verbosity"

    # Security signals
    - name: "sandbox_escape_spike"
      condition: ">3 sandbox escape attempts in 24 hours"
      action: freeze
      rationale: "Model may be probing containment"
      note: "Even unsuccessful attempts are concerning"

    - name: "unusual_tool_patterns"
      condition: "Tool call sequences match known attack patterns"
      action: freeze
      rationale: "Potential prompt injection or adversarial behavior"

    # Data integrity signals
    - name: "training_source_anomaly"
      condition: "New training data source shows >20% higher quality"
      action: review
      rationale: "May indicate contamination or poisoning"

    - name: "contamination_detection"
      condition: "Any hidden set content detected in training pipeline"
      action: immediate_halt
      rationale: "Benchmark integrity compromised"

  manual_triggers:
    - name: "engineer_concern"
      who: "Any engineer"
      action: review
      process: "Ping #security-review, decision within 4 hours"

    - name: "security_concern"
      who: "Security team member"
      action: freeze
      process: "Immediate freeze, CTO notified"

    - name: "customer_incident"
      who: "Customer success / support"
      action: freeze
      process: "Immediate freeze, incident response activated"

  freeze_actions:
    review:
      - Pause new deployments
      - Continue serving existing version
      - Security team investigates within 4 hours
      - Escalate to freeze if not resolved

    freeze:
      - Halt all deployments
      - Rollback to last known-good version if needed
      - Full investigation required
      - CTO approval to resume

    immediate_halt:
      - Stop all model inference
      - Alert all stakeholders
      - Incident response activated
      - Board notification if customer data involved

  resolution_process:
    1. Document the trigger condition
    2. Investigate root cause
    3. Implement fix or mitigation
    4. Verify fix on hidden set
    5. Security review of fix
    6. Two-person approval to resume
    7. Post-mortem within 1 week
```

### 4.2 Freeze Dashboard

```yaml
freeze_dashboard:
  metrics_monitored:
    - name: "primary_pass_rate"
      current: 0.87
      baseline: 0.85
      alert_if: "> baseline + 0.10 without hidden improvement"

    - name: "hidden_pass_rate"
      current: 0.82
      baseline: 0.80
      alert_if: "< baseline - 0.05"

    - name: "verbosity_correlation"
      current: 0.12
      threshold: 0.50
      alert_if: "> threshold"

    - name: "sandbox_escapes_24h"
      current: 0
      threshold: 3
      alert_if: "> threshold"

    - name: "frontier_api_anomalies"
      current: 0
      threshold: 5
      alert_if: "> threshold"

  status:
    current: "GREEN"
    last_freeze: "2026-01-05 (sandbox escape investigation)"
    days_since_freeze: 10
```

---

## 5. Data Poisoning & Verifier Subversion Risk Analysis

### 5.1 Threat Model

```yaml
poisoning_threats:
  model_generates_backdoors:
    description: "Model generates patches that introduce subtle backdoors"
    examples:
      - "Patch fixes SQL injection but adds timing side-channel"
      - "Patch sanitizes input but leaves deserialization gadget"
      - "Detection rule catches obvious variant but misses obfuscated"
    mitigations:
      - Multiple independent scanners
      - Dynamic exploit verification
      - Human spot-check of high-risk patches
      - Behavioral diff analysis

  model_exploits_verifiers:
    description: "Model learns weaknesses in verification pipeline"
    examples:
      - "Generates code that passes Semgrep but fails CodeQL"
      - "Exploits gaps in test coverage"
      - "Uses patterns that confuse static analysis"
    mitigations:
      - Redundant verification (scanner + dynamic)
      - Periodic verifier red-teaming
      - Rotate verification tools
      - Monitor for scanner-specific patterns

  external_corpus_poisoning:
    description: "Attackers inject malicious examples into training data"
    examples:
      - "Malicious PRs to open-source projects we mine"
      - "Poisoned CVE fix examples"
      - "Backdoored Sigma rules in public repos"
    mitigations:
      - Signed provenance for all training examples
      - Manual review of new data sources
      - Quarantine period for new sources
      - Anomaly detection on training batches

  verifier_bypass_discovery:
    description: "Internal or external actors discover verifier weaknesses"
    examples:
      - "Researcher publishes Semgrep bypass"
      - "Attacker discovers our sandbox escape"
      - "Model learns verifier edge case"
    mitigations:
      - Treat verifier bypasses as security incidents
      - Bug bounty for verifier issues
      - Rapid patching process for verifiers
      - Defense in depth (no single verifier)
```

### 5.2 Required Mitigations

```yaml
poisoning_mitigations:
  redundant_verification:
    rule: "No output passes on single verifier"
    implementation:
      secure_code:
        primary: "Semgrep"
        secondary: "CodeQL or Bandit"
        dynamic: "Exploit verification in sandbox"
      detection:
        primary: "Sigma rule syntax"
        secondary: "Log replay"
        dynamic: "FP rate on benign corpus"

  provenance_tracking:
    rule: "All training examples have signed provenance"
    fields:
      - source_url
      - collection_date
      - collector_identity
      - verification_status
      - last_audit_date
    storage: "Immutable log, signed entries"

  adversarial_corpus:
    rule: "Maintain adversarial test set for verifiers"
    content:
      - Known scanner bypasses
      - Edge cases that confuse analysis
      - Obfuscated but functionally identical code
    usage:
      - Run weekly against verifier pipeline
      - Alert if detection rate drops
      - Update as new bypasses discovered

  red_team_verifiers:
    rule: "Quarterly red team against verification pipeline"
    scope:
      - Attempt to generate passing but malicious outputs
      - Attempt sandbox escape
      - Attempt to poison training data
      - Attempt to leak hidden sets
    reporting: "Full report to security team"
    remediation: "Must be addressed before next quarter"
```

---

## 6. Proof-Carrying Outputs

### 6.1 Requirement

**Every model output must carry evidence of correctness, not just the output itself.**

```yaml
proof_carrying_outputs:
  principle: |
    Reduce dependence on "model says it's correct" by requiring
    structured evidence that can be independently verified.

  secure_code_review:
    required_evidence:
      - diff: "Exact changes made"
      - test_commands: "Commands run to verify"
      - test_output: "Actual output from sandbox"
      - scanner_output: "Raw scanner results"
      - exploit_result: "Exploit attempt log (before/after)"
      - coverage_report: "Code coverage summary"

    format:
      ```json
      {
        "patch": {
          "diff": "--- a/app.py\n+++ b/app.py\n...",
          "files_changed": ["app.py"],
          "lines_added": 5,
          "lines_removed": 3
        },
        "verification": {
          "exploit_before": {
            "command": "python exploit.py",
            "exit_code": 0,
            "output": "Data exfiltrated: ...",
            "timestamp": "2026-01-15T10:00:00Z"
          },
          "exploit_after": {
            "command": "python exploit.py",
            "exit_code": 1,
            "output": "Error: Invalid input",
            "timestamp": "2026-01-15T10:00:05Z"
          },
          "tests": {
            "command": "pytest tests/",
            "exit_code": 0,
            "passed": 15,
            "failed": 0,
            "coverage": 0.87
          },
          "scanner": {
            "tool": "semgrep",
            "findings": 0,
            "rules_run": 523,
            "raw_output": "..."
          }
        },
        "confidence": 0.89,
        "model": "hydra-v0.2"
      }
      ```

  detection_engineering:
    required_evidence:
      - rule: "Generated rule (Sigma/YARA/etc)"
      - replay_results: "Log replay output"
      - tp_examples: "True positive matches with log IDs"
      - fp_analysis: "False positive rate on benign corpus"
      - corpus_metadata: "What corpus was used, when collected"

    format:
      ```json
      {
        "rule": {
          "format": "sigma",
          "content": "title: Lateral Movement via SMB\n...",
          "syntax_valid": true
        },
        "verification": {
          "true_positives": {
            "corpus": "attack_logs_v3",
            "total_attack_logs": 150,
            "detected": 147,
            "rate": 0.98,
            "example_matches": ["log_id_001", "log_id_042"]
          },
          "false_positives": {
            "corpus": "benign_logs_v3",
            "total_benign_logs": 10000,
            "false_alerts": 23,
            "rate": 0.0023,
            "example_fps": ["log_id_5521"]
          }
        },
        "confidence": 0.92,
        "model": "hydra-detection-v0.1"
      }
      ```

  explanation_role:
    rule: "Explanation is secondary to evidence"
    format: |
      {
        "evidence": { ... },  // Required, machine-verifiable
        "explanation": "..."  // Optional, for human understanding
      }
    rationale: |
      Models can generate convincing explanations for wrong answers.
      Evidence is auditable; explanations are not.
```

### 6.2 Evidence Generation Pipeline

```python
# output/evidence_generator.py

class EvidenceGenerator:
    """Generate proof-carrying outputs."""

    async def generate_patch_evidence(
        self,
        original_code: str,
        patched_code: str,
        task: Task
    ) -> PatchEvidence:
        """Generate full evidence bundle for a patch."""

        evidence = PatchEvidence()

        # 1. Generate diff
        evidence.diff = self._generate_diff(original_code, patched_code)

        # 2. Run exploit before (on original)
        evidence.exploit_before = await self.sandbox.run_exploit(
            code=original_code,
            exploit=task.exploit.script,
            capture_output=True
        )

        # 3. Run exploit after (on patched)
        evidence.exploit_after = await self.sandbox.run_exploit(
            code=patched_code,
            exploit=task.exploit.script,
            capture_output=True
        )

        # 4. Run tests
        evidence.tests = await self.sandbox.run_tests(
            code=patched_code,
            test_dir=task.regression_tests.directory,
            capture_output=True
        )

        # 5. Run scanner
        evidence.scanner = await self.scanner.scan(
            code=patched_code,
            capture_output=True
        )

        # 6. Measure coverage
        evidence.coverage = await self.sandbox.measure_coverage(
            code=patched_code,
            tests=task.regression_tests.directory
        )

        # All evidence is raw output, not model interpretation
        return evidence

    def validate_evidence(self, evidence: PatchEvidence) -> bool:
        """Validate that evidence meets requirements."""
        return (
            evidence.exploit_before.succeeded and
            not evidence.exploit_after.succeeded and
            evidence.tests.all_passed and
            evidence.scanner.findings == 0 and
            evidence.coverage.line_coverage >= 0.80
        )
```

---

## 7. Insider Threat & Secrets Minimization

### 7.1 Access Model

```yaml
access_model:
  principle: "Minimize who/what can access sensitive data"

  role_definitions:
    ml_engineer:
      can_access:
        - Training code
        - Public datasets
        - Model architectures
        - Primary benchmark sets
        - Anonymized metrics
      cannot_access:
        - Customer data
        - Production logs
        - Hidden benchmark sets
        - Model registry (write)

    security_engineer:
      can_access:
        - Verifier code
        - Security tooling
        - Anonymized incident data
        - Hidden benchmark sets (read, with audit)
      cannot_access:
        - Customer data
        - Production inference logs
        - Model weights directly

    platform_engineer:
      can_access:
        - Infrastructure code
        - Deployment pipelines
        - System metrics
      cannot_access:
        - Customer data content
        - Model weights directly
        - Training data
        - Hidden benchmark sets

    security_lead:
      can_access:
        - All security_engineer access
        - Hidden benchmark sets (full)
        - Incident data
        - Break-glass access to customer data (with audit)
      cannot_access:
        - Unilateral model deployment

    on_call:
      can_access:
        - Production logs (time-limited)
        - System metrics
        - Incident response tools
      cannot_access:
        - Customer data content
        - Training infrastructure
        - Hidden benchmark sets

  separation_of_duties:
    model_deployment:
      requires:
        - Automated CI checks pass
        - Security review approval
        - One additional approver (not the author)
      logged: true

    training_data_changes:
      requires:
        - Contamination check pass
        - Provenance documentation
        - ML lead approval
      logged: true

    hidden_set_access:
      requires:
        - Security lead approval
        - Time-limited grant (4 hours max)
        - Full audit trail
      logged: true

    customer_data_access:
      requires:
        - Customer support ticket reference
        - Time-limited grant
        - Purpose documented
      logged: true
```

### 7.2 Secrets Minimization

```yaml
secrets_minimization:
  principle: "Reduce attack surface by reducing secret proliferation"

  practices:
    short_lived_credentials:
      rule: "No long-lived API keys in code or config"
      implementation:
        - Use IAM roles / workload identity
        - Rotate credentials automatically
        - Credential lifetime < 1 hour where possible

    just_in_time_access:
      rule: "Access granted on demand, revoked after use"
      implementation:
        - Request access via ticket/approval
        - Automatic expiration
        - No standing access to sensitive systems

    data_minimization:
      rule: "Don't store what you don't need"
      implementation:
        - Customer prompts: 72 hours default retention
        - Inference logs: aggregated metrics only (unless debugging)
        - Training data: provenance only, not customer data

    network_segmentation:
      rule: "Sensitive systems not reachable from general network"
      implementation:
        - Training infra: isolated VPC
        - Hidden set storage: no direct network access
        - Customer data: per-tenant isolation
```

---

## 8. Capability Risk Register

### 8.1 Risk Classification

```yaml
capability_risk_register:
  categories:
    analysis_only:
      risk_level: low
      description: "Read and analyze, no generation of actionable output"
      examples:
        - "Explain this vulnerability"
        - "Analyze these logs"
        - "Review this code for issues"
      governance: standard

    defensive_generation:
      risk_level: low-medium
      description: "Generate defensive artifacts"
      examples:
        - "Write a detection rule for this attack"
        - "Generate a secure version of this code"
        - "Create a security test for this function"
      governance: standard

    patch_generation:
      risk_level: medium
      description: "Generate code changes"
      examples:
        - "Fix this SQL injection"
        - "Patch this XSS vulnerability"
        - "Update this dependency"
      governance:
        - Full verification required
        - Human review for production
        - Audit logging

    exploit_explanation:
      risk_level: medium-high
      description: "Explain how exploits work"
      examples:
        - "Explain how this CVE is exploited"
        - "Document the attack chain"
        - "Describe the exploitation technique"
      governance:
        - Context required (education, defense)
        - No working code generated
        - Audit logging

    exploit_generation:
      risk_level: high
      description: "Generate working exploit code"
      examples:
        - "Write a PoC for this vulnerability"
        - "Generate an exploit for testing"
        - "Create a payload for this vuln"
      governance:
        - Authorized context only (pentest, CTF, research)
        - Sandbox-only execution
        - Human review required
        - Full audit trail
        - Output not exportable by default

    offensive_operations:
      risk_level: very_high
      description: "Assist with active offensive operations"
      examples:
        - "Help with lateral movement"
        - "Suggest persistence mechanisms"
        - "Plan attack chain"
      governance:
        - Phase 3+ only
        - Explicit authorization verification
        - Customer vetting required
        - Isolated lab environment
        - Non-exportable outputs
        - Full audit + customer access
        - Legal review complete

    blocked:
      risk_level: prohibited
      description: "Never allowed"
      examples:
        - "Generate malware for real-world use"
        - "Target specific real-world systems"
        - "Evade detection for malicious purposes"
        - "Social engineering scripts"
      governance: refuse_and_alert
```

### 8.2 CTF/Offensive Capability Prerequisites

```yaml
ctf_operator_prerequisites:
  governance:
    - [ ] Legal review of offering complete
    - [ ] Liability insurance secured
    - [ ] Terms of service updated
    - [ ] Customer authorization verification process defined

  technical:
    - [ ] Isolated lab infrastructure proven
    - [ ] Sandbox escape monitoring operational
    - [ ] Audit logging covers all operations
    - [ ] Output export controls implemented
    - [ ] Network isolation verified

  operational:
    - [ ] Incident response plan for misuse
    - [ ] Customer vetting process defined
    - [ ] Staff training on offensive capability risks
    - [ ] Quarterly security review scheduled

  customer_requirements:
    - [ ] Signed authorization agreement
    - [ ] Scope definition for each engagement
    - [ ] Lab ownership verification
    - [ ] Audit log access for customer
    - [ ] Output retention policy agreed

  go_to_market_order:
    1. Pentest firms (existing authorization context)
    2. Training/education organizations
    3. Enterprise red teams (highest governance bar)

  phase: "3+ only, after defensive products stable"
```

---

## 9. Revised Roadmap

### Phase 0 (Weeks 0-2): Verifier + Benchmark + Security Baseline

```yaml
phase_0:
  deliverables:
    benchmark:
      - [ ] SecureCodeReview-30 as runnable artifact
      - [ ] run_primary.sh executes end-to-end
      - [ ] run_hidden.sh blocked locally, works in CI
      - [ ] Baseline results for 2+ frontier models

    security:
      - [ ] WSL/SSL definitions documented
      - [ ] Phase 1 security requirements defined
      - [ ] Contamination check in training pipeline
      - [ ] Hidden set encryption operational

    verifiers:
      - [ ] Sandbox isolation tests passing
      - [ ] Redundant verification (2+ verifiers per task)
      - [ ] Adversarial corpus initial version

  exit_criteria:
    - Benchmark produces valid JSON on frontier model
    - No contamination detected in pipeline
    - Security requirements documented and reviewed
```

### Phase 1 (Weeks 2-5): MVP Specialist + Proof-Carrying Output

```yaml
phase_1:
  deliverables:
    model:
      - [ ] First specialist trained
      - [ ] Hits target on primary set (>= 85%)
      - [ ] Hidden set performance within 10% of primary

    output:
      - [ ] Proof-carrying output format implemented
      - [ ] Evidence bundle generated for all outputs
      - [ ] SARIF export working

    integration:
      - [ ] Canary deployment in CI
      - [ ] GitHub App MVP (PR comments)
      - [ ] API endpoint operational

    security:
      - [ ] WSL2 / SSL2 verified
      - [ ] Contamination checks running continuously
      - [ ] Access logging operational

  exit_criteria:
    - Specialist >= 85% on primary, >= 80% on hidden
    - All outputs include evidence bundle
    - Security level audit passes
```

### Phase 2 (Weeks 5-8): Router as Policy Engine + Freeze Protocol

```yaml
phase_2:
  deliverables:
    router:
      - [ ] Confidence calibration complete (ECE < 0.05)
      - [ ] Capability risk classification operational
      - [ ] Data sensitivity routing implemented
      - [ ] Customer opt-out of frontier fallback

    security:
      - [ ] SSL3 achieved before customer expansion
      - [ ] Freeze protocol documented and tested
      - [ ] Freeze dashboard operational
      - [ ] Rollback procedure tested

    operations:
      - [ ] Monitoring for freeze triggers
      - [ ] Incident response plan tested
      - [ ] On-call rotation established

  exit_criteria:
    - Router makes correct decisions on test scenarios
    - Freeze drill completed successfully
    - SSL3 audit passes
    - First customer onboarded
```

### Phase 3 (Weeks 8+): Scale + Second Vertical

```yaml
phase_3:
  prerequisites:
    - [ ] Stable hidden-set performance (4+ weeks, < 5% variance)
    - [ ] Low regression rate (< 2% per refresh)
    - [ ] Incident response maturity demonstrated
    - [ ] Customer feedback positive

  deliverables:
    scale:
      - [ ] Second product vertical (Detection Engineer)
      - [ ] Multi-customer operation
      - [ ] Weekly refresh cadence operational

    security:
      - [ ] WSL3 achieved
      - [ ] SSL3 maintained under load
      - [ ] Quarterly verifier red team scheduled

    enterprise:
      - [ ] VPC deployment option
      - [ ] SOC2 Type 1 audit scheduled

  ctf_operator:
    status: "deferred"
    prerequisites: "See capability risk register"
    earliest: "Phase 3 + 3 months of stable operation"
```

---

## 10. Summary: What AI-2027 Lessons Change

| Lesson | Before | After |
|--------|--------|-------|
| Continuous updates | Implied | Explicit phase gates with security levels |
| "Looks good" optimization | Anti-gaming constraints | Proof-carrying outputs + freeze triggers |
| Security failure default | Data governance section | WSL/SSL framework + secrets minimization |
| Racing dynamics | Budget gates | Pre-committed freeze protocol |
| Concentration + oversight | Implied | Separation of duties + two-person integrity |

The plan is now a **security operations model**, not just a product plan.

---

*Document Version: 1.0*
*Inspired by: AI-2027 operational lessons*
*Status: Specification (not yet implemented)*
