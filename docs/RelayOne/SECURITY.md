# Security Configuration Guide

## Overview

relay.one is an enterprise AI agent governance platform with comprehensive security features. This document covers security configuration, best practices, and compliance considerations.

---

## Table of Contents

- [Required Secrets](#required-secrets)
- [Authentication](#authentication)
- [Authorization](#authorization)
- [Encryption](#encryption)
- [Network Security](#network-security)
- [Rate Limiting](#rate-limiting)
- [Audit Logging](#audit-logging)
- [Certificate Management](#certificate-management)
- [Quantum-Safe Cryptography](#quantum-safe-cryptography)
- [Deployment Security](#deployment-security)
- [Compliance](#compliance)
- [Incident Response](#incident-response)

---

## Required Secrets

**Never commit secrets to version control.**

### JWT_SECRET

JSON Web Token secret for authentication.

| Requirement | Value |
|-------------|-------|
| Minimum length | 32 characters |
| Type | Cryptographically random |
| Scope | Unique per environment |

**Generate:**
```bash
openssl rand -base64 48
```

### ENCRYPTION_KEY

Encryption key for sensitive data at rest.

| Requirement | Value |
|-------------|-------|
| Length | 32 bytes (64 hex chars) |
| Type | Cryptographically random |
| Scope | Unique per environment |

**Generate:**
```bash
openssl rand -hex 32
```

**WARNING:** Loss of this key means permanent data loss for encrypted fields.

### WORKFLOW_ENCRYPTION_KEY

Encryption key for workflow credentials.

```bash
openssl rand -hex 32
```

---

## Authentication

### JWT Tokens

| Setting | Default | Description |
|---------|---------|-------------|
| Algorithm | HS256 | HMAC SHA-256 |
| Expiry | 24h | Token lifetime |
| Refresh | 7d | Refresh token lifetime |

**Configuration:**
```bash
JWT_SECRET=your-secret
JWT_ISSUER=relay.one
JWT_EXPIRY=24h
```

### Multi-Factor Authentication

MFA is available for all users and can be enforced organization-wide.

**Organization Enforcement:**
```typescript
// In organization settings
{
  enforceMfa: true
}
```

**Supported Methods:**
- TOTP (Google Authenticator, Authy)
- Backup codes (one-time use)

### API Key Authentication

API keys provide programmatic access with scoped permissions.

**Security Features:**
- Keys are hashed (SHA-256) before storage
- Only prefix shown after creation
- IP allowlist support
- Expiration dates
- Request rate limits

**Scopes:**
```
agents:read        # Read agent data
agents:write       # Create/update agents
governance:read    # Read policies
governance:write   # Manage policies
billing:read       # View billing
billing:write      # Manage billing
admin:*           # Full admin access
```

### SSO Integration

**SAML 2.0:**
- Configure IdP metadata
- Map attributes to user fields
- Enforce SSO for domain

**OIDC/OAuth 2.0:**
- Supported providers: Google, Microsoft, Okta, Auth0
- Custom OIDC configuration

---

## Authorization

### Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| `owner` | Full organization control |
| `admin` | Manage users, agents, policies |
| `member` | Create agents, view policies |
| `viewer` | Read-only access |

### Capability-Based Access

Agents request capabilities, users grant them:

```typescript
const capabilities = [
  'file-read',
  'file-write',
  'web-search',
  'code-execution',
  'payment-send',
  'pii-access'
];
```

**High-Risk Capabilities:**
- `payment-send` - Requires HITL approval
- `pii-access` - Logged and monitored
- `code-execution` - Sandboxed environments

### Policy Enforcement

Governance policies control agent behavior:

| Policy Type | Description |
|-------------|-------------|
| ACL | Allowlist/blocklist rules |
| PII | PII detection and redaction |
| Rate Limit | Request rate limiting |
| Spending | Budget limits |
| HITL | Require human approval |
| Content | Content filtering |

---

## Encryption

### Data at Rest

| Data Type | Encryption |
|-----------|------------|
| Passwords | Bcrypt (cost 12) |
| API keys | SHA-256 hash |
| MFA secrets | AES-256-GCM |
| Workflow credentials | AES-256-GCM |
| Private keys | AES-256-GCM |

### Data in Transit

- TLS 1.2+ required for all connections
- HSTS enabled
- Certificate pinning for critical services

### Database Encryption

**MongoDB:**
```bash
# Enable encryption at rest
mongod --enableEncryption --encryptionKeyFile /path/to/keyfile
```

**Field-Level Encryption:**
Sensitive fields are encrypted at application level before storage.

---

## Network Security

### CORS Configuration

```bash
# Production - specify exact origins
CORS_ORIGINS=https://console.relay.one,https://admin.relay.one

# Never use * in production
```

### IP Allowlisting

Configure per organization:
```typescript
{
  settings: {
    ipAllowlist: ['10.0.0.0/8', '192.168.1.0/24']
  }
}
```

### Firewall Rules

**Required Ports:**

| Port | Service | Access |
|------|---------|--------|
| 3001 | API | Load balancer only |
| 3100 | Gateway | Load balancer only |
| 27017 | MongoDB | Internal only |
| 6379 | Redis | Internal only |

### DDoS Protection

- Use cloud provider DDoS protection (AWS Shield, Cloudflare)
- Configure rate limiting at CDN/load balancer level
- Enable geographic restrictions if applicable

---

## Rate Limiting

### Global Limits

```bash
RATE_LIMIT_WINDOW=60000      # 1 minute
RATE_LIMIT_MAX_TOKENS=100    # requests per window
```

### Per-Endpoint Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/auth/login` | 10 | 15 min |
| `/api/v1/agents` | 100 | 1 min |
| `/gateway/invoke` | 1000 | 1 min |

### Per-API-Key Limits

```typescript
{
  restrictions: {
    maxRequestsPerDay: 10000
  }
}
```

---

## Audit Logging

### What's Logged

| Category | Events |
|----------|--------|
| Auth | Login, logout, password reset, MFA |
| Agents | Create, update, delete, invoke |
| Policies | Create, update, enable/disable |
| Governance | Violations, approvals, blocks |
| Billing | Subscription changes, payments |
| Admin | User management, settings changes |

### Log Format

```json
{
  "action": "agent.create",
  "category": "agent",
  "actor": {
    "type": "user",
    "id": "user-123",
    "email": "user@example.com",
    "ip": "192.168.1.1"
  },
  "target": {
    "type": "Agent",
    "id": "agent-456",
    "name": "MyAgent"
  },
  "result": "success",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Retention

```bash
# Configure TTL on audit_logs collection
# Default: 90 days
```

### Compliance Export

Export audit logs for compliance:
```bash
GET /api/v1/admin/audit-logs/export?start=2024-01-01&end=2024-01-31
```

---

## Certificate Management

### mTLS Certificates

relay.one issues and manages mTLS certificates for secure agent communication.

**Certificate Types:**
- Organization certificates
- Agent certificates
- Deployment certificates

**Lifecycle:**
```
Request → Pending → Approved → Active → Renewal/Expiry
                 ↓
              Rejected
```

### Certificate Authority

```bash
# Use central CA
RELAY_CA_URL=https://ca.relay.one
RELAY_CA_API_KEY=your-api-key

# Or self-hosted CA
ROOT_CA_CERTIFICATE="-----BEGIN CERTIFICATE-----\n..."
ROOT_CA_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."
```

### Auto-Renewal

```bash
CERT_AUTO_RENEW=true
KEY_ROTATION_DAYS=90
```

---

## Quantum-Safe Cryptography

relay.one includes post-quantum cryptography features for forward security.

### Enable Quantum-Safe Mode

```bash
QUANTUM_SAFE_MODE=true
FEATURE_QUANTUM_SAFE=true
```

### Threat Detection

The platform detects "Harvest Now, Decrypt Later" attacks:
- Monitors for unusual data exfiltration patterns
- Alerts on suspicious bulk data access
- Prioritizes quantum-vulnerable data protection

### Algorithms

When quantum-safe mode is enabled:
- Key exchange: Kyber (NIST-approved)
- Signatures: Dilithium (NIST-approved)
- Hybrid mode: Classical + post-quantum

---

## Deployment Security

### Docker Security

```yaml
# docker-compose.yml security settings
services:
  api:
    security_opt:
      - no-new-privileges:true
    read_only: true
    cap_drop:
      - ALL
    user: "1000:1000"
```

### Kubernetes Security

```yaml
# Pod security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

### Secret Management

**Recommended:**
- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- HashiCorp Vault

```bash
# Create Kubernetes secret
kubectl create secret generic relay-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 48) \
  --from-literal=encryption-key=$(openssl rand -hex 32)
```

### Image Scanning

Scan container images for vulnerabilities:
```bash
trivy image relay-one/api:latest
```

---

## Compliance

### SOC 2

relay.one supports SOC 2 compliance:

| Control | Implementation |
|---------|---------------|
| CC6.1 | Logical access controls (RBAC) |
| CC6.6 | Audit logging |
| CC6.7 | Encryption in transit/at rest |
| CC7.2 | Incident detection |

### HIPAA

For healthcare deployments:

| Requirement | Feature |
|-------------|---------|
| Access Controls | RBAC + MFA enforcement |
| Audit Logs | Immutable audit trail |
| Encryption | AES-256 for PHI |
| BAA | Available for Enterprise |

### GDPR

Data protection features:

| Right | Implementation |
|-------|---------------|
| Access | Data export API |
| Rectification | User data update |
| Erasure | Account deletion |
| Portability | Standard export formats |

### PCI DSS

For payment data:
- No cardholder data storage
- Stripe handles payment processing
- Webhook signature verification

---

## Incident Response

### If Secrets Are Compromised

**Immediate (0-15 minutes):**

1. Rotate all affected secrets:
```bash
# Generate new secrets
NEW_JWT=$(openssl rand -base64 48)
NEW_ENCRYPTION=$(openssl rand -hex 32)

# Update environment
kubectl set env deployment/api \
  JWT_SECRET=$NEW_JWT \
  ENCRYPTION_KEY=$NEW_ENCRYPTION
```

2. Revoke all API keys
3. Force logout all sessions
4. Enable enhanced logging

**Investigation (15-60 minutes):**

1. Review audit logs for unauthorized access
2. Identify scope of exposure
3. Check for data exfiltration
4. Document timeline

**Remediation (1-24 hours):**

1. Patch vulnerability
2. Update all dependent systems
3. Re-encrypt affected data
4. Notify affected users if required

### Security Contacts

- Security issues: security@relay.one
- Bug bounty: https://relay.one/security
- Emergency: [On-call rotation]

---

## Security Checklist

### Pre-Deployment

- [ ] Generate unique secrets for environment
- [ ] Configure TLS certificates
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Configure rate limiting
- [ ] Set up monitoring/alerting
- [ ] Review CORS settings
- [ ] Configure backup encryption

### Post-Deployment

- [ ] Verify TLS configuration (SSL Labs)
- [ ] Test authentication flows
- [ ] Verify audit logs are recording
- [ ] Test rate limiting
- [ ] Run security scan
- [ ] Document emergency procedures

### Ongoing

- [ ] Rotate secrets quarterly
- [ ] Review access logs monthly
- [ ] Update dependencies weekly
- [ ] Security training annually
- [ ] Penetration testing annually
- [ ] Disaster recovery testing

---

## See Also

- [ENVIRONMENT.md](ENVIRONMENT.md) - Environment variables
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment instructions
- [docs/architecture/iam.md](docs/architecture/iam.md) - IAM details
- [docs/architecture/governance.md](docs/architecture/governance.md) - Governance policies
