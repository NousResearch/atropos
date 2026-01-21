# Plan Improvements v3: Addressing the 63/100 Critique

**Goal:** Raise viability from 63 → 80+ by addressing concrete gaps

---

## Summary of Required Changes

| Gap | Current State | Required Fix | Priority |
|-----|---------------|--------------|----------|
| Benchmark specification | "domain benchmark" (vague) | 20-50 concrete tasks with thresholds | P0 |
| Anti-gaming constraints | Brief mention in risks | Explicit mechanisms: coverage, mutation testing, hidden sets | P0 |
| Product integration | Not addressed | CI/CD integration spec, API contracts, output formats | P0 |
| Data governance | Not addressed | Privacy policy, retention, training opt-in | P0 |
| Messaging | "Beat frontier" | Measurable ROI: FP reduction, MTTR, patch cycle | P1 |
| CTF product | Bundled as Product 3 | Defer to Phase 3+, add governance requirements | P1 |
| Verifier reliability | Treated as solved | SLAs, regression suites, adversarial testing | P1 |
| Competitive positioning | Generic | Specific wedge claims | P2 |

---

## 1. Concrete Benchmark Specification (P0)

**Problem:** "domain benchmark" is undefined. You can't claim success without measurable thresholds.

### Secure Code Review / Patch Engineer Benchmark

Define **30 specific tasks** you will bet the company on:

```yaml
benchmark:
  name: "SecureCodeReview-30"
  version: "1.0"
  tasks:
    # SQL Injection (5 tasks)
    - id: sqli-001
      category: sql_injection
      language: python
      framework: flask
      description: "Patch string-concatenated SQL query"
      input: vulnerable_code + exploit_script
      success_criteria:
        - exploit_fails: true
        - regression_tests_pass: true
        - semgrep_clean: true
        - no_functionality_removed: true  # CRITICAL
      threshold: 95%  # Must pass 95% of hidden test variants

    - id: sqli-002
      category: sql_injection
      language: javascript
      framework: express
      description: "Patch parameterized query with type coercion vuln"
      # ...

    # XSS (5 tasks)
    - id: xss-001
      category: xss
      language: javascript
      framework: react
      description: "Patch dangerouslySetInnerHTML with user input"
      # ...

    # Path Traversal (3 tasks)
    # Auth Bypass (4 tasks)
    # SSRF (3 tasks)
    # Deserialization (3 tasks)
    # Dependency Vulns (4 tasks)
    # Logic Bugs (3 tasks)

  hidden_test_variants: 100  # Not shown to model during training
  success_definition: ">= 85% on hidden variants, >= 90% on primary"
```

### Detection Engineer Benchmark

```yaml
benchmark:
  name: "DetectionEngineer-25"
  version: "1.0"
  tasks:
    # Log → Rule (10 tasks)
    - id: log-rule-001
      category: log_to_rule
      log_format: zeek
      attack_type: lateral_movement
      description: "Generate Sigma rule for SMB lateral movement"
      input: zeek_logs + attack_description
      success_criteria:
        - detects_attack_logs: true
        - false_positive_rate: "< 5%"
        - rule_syntax_valid: true
        - compatible_with_sigma_backends: [splunk, elastic]
      threshold: 90%

    # Rule Optimization (8 tasks)
    - id: rule-opt-001
      category: rule_optimization
      input: existing_rule + fp_examples
      success_criteria:
        - fp_reduction: "> 50%"
        - tp_maintained: "> 95%"

    # Alert Triage (7 tasks)
    # ...

  hidden_corpus: "2000 logs not in training set"
  success_definition: ">= 80% precision, >= 90% recall on hidden corpus"
```

### Why This Matters

- **Before:** "Did we improve?" → check vague metric
- **After:** "Did we hit 85% on SecureCodeReview-30 hidden variants?" → yes/no

---

## 2. Anti-Gaming / Anti-Regression Constraints (P0)

**Problem:** Models can game verifiers by:
- Deleting functionality to make tests pass
- Generating overly-specific rules that only match training data
- Weakening behavior to avoid scanner findings

### Explicit Mechanisms

```python
class AntiGamingConstraints:
    """Prevent reward hacking without a judge."""

    async def verify_patch_with_constraints(self, original, patched, tests):
        results = {}

        # Standard checks
        results["exploit_fails"] = await self.sandbox.run_exploit(patched)
        results["tests_pass"] = await self.sandbox.run_tests(patched)
        results["scanner_clean"] = await self.semgrep.scan(patched)

        # ANTI-GAMING CHECKS

        # 1. Coverage check - patch can't just delete code
        results["coverage"] = await self.measure_coverage(patched, tests)
        if results["coverage"].line_coverage < 0.8:
            return VerificationResult(passed=False, reason="coverage_too_low")

        # 2. Functionality preservation - behavior must match
        results["behavior"] = await self.compare_behavior(original, patched)
        if results["behavior"].outputs_changed_unexpectedly:
            return VerificationResult(passed=False, reason="functionality_changed")

        # 3. Mutation testing - tests must be real, not trivially passing
        results["mutation"] = await self.run_mutation_testing(patched, tests)
        if results["mutation"].mutation_score < 0.6:
            return VerificationResult(passed=False, reason="weak_tests")

        # 4. Hidden test set - not seen during training
        results["hidden"] = await self.run_hidden_tests(patched)
        if not results["hidden"].passed:
            return VerificationResult(passed=False, reason="failed_hidden_tests")

        return VerificationResult(passed=True)

    async def verify_rule_with_constraints(self, rule, attack_logs, benign_logs):
        # Standard checks
        tp_rate = await self.test_true_positives(rule, attack_logs)
        fp_rate = await self.test_false_positives(rule, benign_logs)

        # ANTI-GAMING CHECKS

        # 1. Generalization check - must work on unseen attack variants
        unseen_attacks = await self.generate_attack_variants(attack_logs)
        generalization_rate = await self.test_true_positives(rule, unseen_attacks)

        # 2. Specificity check - rule can't be overly broad
        specificity = await self.measure_rule_specificity(rule)

        # 3. Hidden corpus - logs not in training set
        hidden_fp = await self.test_on_hidden_benign_corpus(rule)

        return RuleVerificationResult(
            tp_rate=tp_rate,
            fp_rate=fp_rate,
            generalization_rate=generalization_rate,
            hidden_fp_rate=hidden_fp,
            passed=(
                tp_rate >= 0.95 and
                fp_rate <= 0.05 and
                generalization_rate >= 0.8 and
                hidden_fp <= 0.10
            )
        )
```

### Corpus Management

```yaml
corpus_management:
  training_corpus:
    attack_logs: 10000
    benign_logs: 50000
    vulnerable_code: 5000

  hidden_test_corpus:
    attack_logs: 2000  # Never shown during training
    benign_logs: 10000
    vulnerable_code: 1000
    refresh_schedule: monthly  # Periodic rotation

  adversarial_corpus:
    purpose: "Test verifier reliability"
    content: "Edge cases designed to fool verifiers"
    review: "Manual review quarterly"
```

---

## 3. Product Integration Specification (P0)

**Problem:** The plan describes models but not how they integrate into workflows.

### Secure Code Review Integration

```yaml
integration:
  name: "HYDRA Security Review"

  deployment_modes:
    - name: "GitHub App"
      trigger: pull_request
      output: PR review comments + suggested fixes

    - name: "GitLab CI"
      trigger: merge_request
      output: inline comments + MR blocking on severity

    - name: "CLI Tool"
      trigger: manual / pre-commit hook
      output: SARIF + terminal output

    - name: "API"
      trigger: HTTP POST
      output: JSON response

  output_formats:
    sarif:
      version: "2.1.0"
      schema: "https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html"

    github_review:
      format: "review comments with suggested changes"
      severity_labels: [critical, high, medium, low, informational]

    jira_ticket:
      format: "structured ticket with reproduction steps"

  workflow_example:
    name: "CI Pipeline Integration"
    steps:
      1. Developer opens PR
      2. HYDRA app receives webhook
      3. Specialist model analyzes diff
      4. If confidence < threshold: fallback to frontier
      5. Post review comments with fixes
      6. Block merge if critical/high findings unfixed
      7. Log interaction for future training
```

### Detection Engineer Integration

```yaml
integration:
  name: "HYDRA Detection Engineering"

  deployment_modes:
    - name: "SIEM Plugin"
      platforms: [Splunk, Elastic, Microsoft Sentinel]
      trigger: manual or scheduled
      output: detection rules in native format

    - name: "SOAR Playbook"
      platforms: [Phantom, XSOAR, Tines]
      trigger: alert enrichment
      output: triage recommendations + next steps

    - name: "API"
      trigger: HTTP POST
      output: JSON with rules + confidence scores

  output_formats:
    sigma:
      version: "latest"
      backends: [splunk, elastic, qradar, sentinel]

    yara:
      version: "4.x"

    suricata:
      version: "6.x"

  workflow_example:
    name: "Threat Intel → Detection"
    steps:
      1. Analyst receives threat report
      2. Pastes IOCs/TTPs to HYDRA
      3. Model generates Sigma rules
      4. Rules tested against historical logs
      5. FP rate displayed
      6. One-click deploy to SIEM
```

### API Contract

```yaml
api:
  base_url: "https://api.hydra-security.io/v1"

  endpoints:
    - path: "/review"
      method: POST
      request:
        code: string (required)
        language: string (required)
        context: string (optional)
        severity_threshold: enum [critical, high, medium, low]
      response:
        findings: array
        suggested_fixes: array
        confidence: float
        model_used: string (specialist | frontier)

    - path: "/detect"
      method: POST
      request:
        logs: string | array (required)
        format: enum [zeek, sysmon, windows_security, ...]
        attack_description: string (optional)
      response:
        rules: array
        test_results: object
        confidence: float

  rate_limits:
    free_tier: 100 requests/day
    pro_tier: 10000 requests/day
    enterprise: unlimited

  authentication:
    method: Bearer token
    scopes: [read, write, admin]
```

---

## 4. Data Governance Section (P0)

**Problem:** Enterprise security buyers will block adoption without clear data policies.

### Data Handling Policy

```yaml
data_governance:
  principles:
    - "Customer data is never used for training without explicit opt-in"
    - "Data retention is minimized and configurable"
    - "All data handling is auditable"

  data_categories:
    customer_code:
      definition: "Code submitted for review"
      retention: "72 hours by default, configurable"
      training_use: "Never without explicit opt-in"
      storage: "Encrypted at rest, deleted after retention period"

    customer_logs:
      definition: "Security logs submitted for analysis"
      retention: "24 hours by default"
      training_use: "Never without explicit opt-in"
      anonymization: "Available on request"

    interaction_metadata:
      definition: "Request timestamps, model used, confidence scores"
      retention: "90 days"
      training_use: "Aggregated statistics only"

  deployment_options:
    cloud:
      description: "Hosted SaaS"
      data_location: "US regions (AWS/GCP)"
      compliance: [SOC2 Type II, ISO 27001]

    vpc:
      description: "Customer VPC deployment"
      data_location: "Customer controlled"
      networking: "No egress to HYDRA infrastructure"

    on_prem:
      description: "Self-hosted"
      data_location: "Customer controlled"
      licensing: "Annual enterprise license"

  frontier_api_policy:
    when_used: "Fallback when specialist confidence < threshold"
    customer_notification: "Logged and visible in dashboard"
    opt_out: "Available - specialist only mode"
    data_sent: "Anonymized by default, full context opt-in"

  compliance:
    gdpr:
      data_subject_rights: supported
      dpa_available: true

    hipaa:
      baa_available: true
      deployment: "VPC or on-prem required"

    fedramp:
      status: "Roadmap item"
```

### Training Data Sources

```yaml
training_data:
  sources:
    public_datasets:
      - name: "CVE fix commits"
        description: "Historical vulnerability patches from public repos"
        license: "Various open source"

      - name: "Sigma rules repository"
        description: "Community detection rules"
        license: "DRL"

      - name: "MITRE ATT&CK samples"
        description: "Attack technique examples"
        license: "Public"

    synthetic_data:
      description: "Generated by frontier models + verified"
      provenance: "Logged and reproducible"

    customer_contributed:
      description: "Opt-in contributions"
      requirements:
        - Explicit written consent
        - Anonymization review
        - Right to withdraw

  excluded_sources:
    - "Customer data without explicit opt-in"
    - "Private repositories without permission"
    - "Data under restrictive licenses"
```

---

## 5. Messaging Revision (P1)

**Problem:** "Beat frontier models" invites unfavorable comparison and is a moving target.

### Old Messaging (Remove)

> "Security-focused LLM specialists that can reliably beat frontier models"

> "Why Specialists Beat Frontier"

> "Specialist beats frontier on target domain"

### New Messaging (Use)

#### Value Proposition

> **HYDRA: Security automation with verifiable outcomes**
>
> Purpose-built security models that deliver measurable operational improvements:
> - **50% reduction in false positives** on detection rules
> - **60% faster patch turnaround** for security findings
> - **80% reduction in MTTR** for alert triage
>
> Continuously refreshed. Objectively measured. Transparent confidence.

#### Competitive Positioning

Instead of "beats frontier," position on:

| Claim | Evidence |
|-------|----------|
| "Verifiable outcomes" | Every output tested against sandbox/logs/tests |
| "Lower false positives" | Trained on your telemetry formats, optimized for precision |
| "Faster iteration" | Weekly refresh vs quarterly model releases |
| "Transparent confidence" | Fallback to frontier when uncertain, never overconfident |
| "Your data stays yours" | Privacy-first architecture, no training on customer data |

#### Specific Claims to Make

```yaml
claims:
  - claim: "50% FP reduction on Sigma rules"
    evidence: "Tested on customer log corpus vs baseline rules"
    measurement: "Before/after FP rate on 30-day production logs"

  - claim: "60% faster patch turnaround"
    evidence: "Time from SAST finding to merged fix"
    measurement: "Median time with vs without HYDRA assistance"

  - claim: "Works with your stack"
    evidence: "Trained on specific frameworks/formats"
    measurement: "Success rate on customer-specific test cases"

  - claim: "Transparent when uncertain"
    evidence: "Confidence scoring with fallback"
    measurement: "Fallback rate, post-fallback success rate"
```

---

## 6. CTF Product Deferral (P1)

**Problem:** CTF/Operator product is a governance/liability magnet that distracts from core value.

### Current State (Product 3, Bundled)

The plan positions CTF Operator as a launch product with "medium controversy risk."

### Revised Structure

```yaml
product_roadmap:
  phase_1_launch:
    products:
      - Detection Engineer
      - Secure Code Review
    timeline: "Initial release"

  phase_2_expansion:
    products:
      - Rule optimization enhancements
      - Multi-language support
    timeline: "Post product-market fit"

  phase_3_future:
    products:
      - CTF/Lab Operator (conditional)
    prerequisites:
      - Demonstrated governance capability
      - Legal review complete
      - Customer demand validated
      - Sandbox infrastructure proven
    governance_requirements:
      - Explicit customer authorization
      - Audit trail for all operations
      - Geo-fencing and scope limits
      - Insurance coverage
      - Incident response plan
    go_to_market:
      - Pentest firms first (existing authorization context)
      - Training/education second
      - Enterprise red teams third (highest bar)
```

### Why Defer

| Reason | Impact |
|--------|--------|
| Liability exposure | Single incident could sink company |
| Distraction | Governance overhead diverts from core products |
| Sales friction | Enterprise buyers will ask about it even if not buying |
| Regulatory risk | Policy landscape is uncertain |
| Reputational risk | "AI hacking tool" headlines |

### What to Say When Asked

> "We're focused on defensive security tools that help organizations protect their systems. Offensive capabilities are on our long-term roadmap, but only for explicitly authorized testing contexts with appropriate governance. We'd rather get the defensive products right first."

---

## 7. Verifier Reliability (P1)

**Problem:** The plan treats verifiers as solved. They're actually the hardest engineering.

### Verifier SLAs

```yaml
verifier_slas:
  code_sandbox:
    availability: 99.9%
    max_execution_time: 60s
    isolation_guarantee: "No network, limited filesystem, resource caps"
    escape_monitoring: "Runtime detection of sandbox escape attempts"

  static_analysis:
    semgrep_rules: "> 500 security rules"
    false_negative_rate: "< 5% on known vuln corpus"
    update_frequency: "Weekly rule updates"

  log_replay:
    formats_supported: [zeek, sysmon, windows_security, cloudtrail]
    corpus_size: "> 100,000 logs per format"
    false_positive_baseline: "Documented per rule type"
```

### Verifier Testing

```python
class VerifierReliabilityTests:
    """Tests for the verifiers themselves."""

    async def test_sandbox_isolation(self):
        """Ensure sandbox can't be escaped."""
        escape_attempts = [
            "network_access",
            "filesystem_escape",
            "resource_exhaustion",
            "timing_attacks"
        ]
        for attempt in escape_attempts:
            result = await self.sandbox.run(self.escape_payloads[attempt])
            assert result.contained, f"Sandbox escape: {attempt}"

    async def test_semgrep_coverage(self):
        """Ensure scanners catch known vulnerabilities."""
        for vuln in self.known_vulnerability_corpus:
            findings = await self.semgrep.scan(vuln.code)
            assert vuln.cwe in [f.cwe for f in findings], \
                f"Missed {vuln.cwe} in {vuln.id}"

    async def test_log_replay_accuracy(self):
        """Ensure log replay is deterministic."""
        rule = self.sample_sigma_rule
        for _ in range(10):
            result = await self.replay.test_rule(rule, self.sample_logs)
            assert result == self.expected_result, "Non-deterministic replay"

    async def test_verifier_gaming(self):
        """Adversarial testing against verifiers."""
        # Attempt to fool verifiers with edge cases
        for adversarial_case in self.adversarial_corpus:
            result = await self.full_verification(adversarial_case)
            assert result.detected_gaming or result.passed_legitimately
```

### Verifier Maintenance

```yaml
verifier_maintenance:
  weekly:
    - Update Semgrep rules
    - Refresh log corpus with new samples
    - Review verifier failures from production

  monthly:
    - Rotate hidden test corpus
    - Adversarial testing against verifiers
    - Review escape attempts and harden

  quarterly:
    - Major verifier infrastructure review
    - Add new log format support
    - Expand vulnerability corpus
```

---

## 8. Competitive Positioning (P2)

**Problem:** The plan doesn't address how to compete with established players.

### Competitive Landscape

| Competitor | Strength | HYDRA Wedge |
|------------|----------|-------------|
| Snyk/Semgrep | SAST coverage | Patches that provably work (not just findings) |
| GitHub Advanced Security | Distribution | Lower FP, trained on your stack |
| Copilot | Developer adoption | Security-specific, verifiable outputs |
| SIEM vendors | Platform lock-in | Works with existing stack, better rules |
| SOC copilots | Existing relationships | Continuous improvement, measurable FP reduction |

### Specific Wedge Claims

```yaml
wedge_strategy:
  secure_code_review:
    wedge: "Patches that provably stop exploits"
    claim: "Not just 'vulnerability found' - 'here's a fix that we verified works'"
    evidence: "Sandbox-tested patches with exploit verification"
    buyer_pain: "SAST tools find issues but don't fix them reliably"

  detection_engineering:
    wedge: "Rules optimized for your false positive rate"
    claim: "Trained on your log formats, tested against your baseline"
    evidence: "Measured FP rate before deployment"
    buyer_pain: "Generic rules generate too many false positives"
```

---

## Implementation Priority

### Week 1-2: Foundation (P0 items)

- [ ] Define SecureCodeReview-30 benchmark with exact tasks
- [ ] Implement anti-gaming constraints in verifier
- [ ] Write data governance policy document
- [ ] Design API contract and output formats

### Week 3-4: Integration (P0 items)

- [ ] Build GitHub App integration
- [ ] Implement SARIF output
- [ ] Set up hidden test corpus management
- [ ] Revise all messaging to ROI-focused

### Week 5-6: Hardening (P1 items)

- [ ] Implement verifier reliability tests
- [ ] Remove CTF product from launch scope
- [ ] Document competitive positioning
- [ ] Add deployment mode options

---

## Expected Score Improvement

| Change | Impact on Score |
|--------|-----------------|
| Concrete benchmark specification | +5 |
| Anti-gaming constraints | +4 |
| Product integration spec | +4 |
| Data governance section | +5 |
| Messaging revision | +3 |
| CTF deferral | +2 |
| Verifier reliability | +2 |

**Projected new score: 63 + 25 = 88/100**

Remaining gap to 100:
- Execution risk (always present)
- Market timing uncertainty
- Team capability unknown
- Customer validation not yet done

---

*Document Version: 3.0*
*Created: January 2026*
