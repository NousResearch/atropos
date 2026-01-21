# relay.one Environment Variables

Complete reference for all environment variables used by the platform.

## Quick Start

```bash
# Copy the example file
cp deploy/.env.example .env

# Required minimum variables
MONGODB_URI=mongodb://localhost:27017/relay_one
JWT_SECRET=your-secure-jwt-secret-at-least-32-chars
INSTANCE_ID=platform-1
```

---

## Core Platform

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NODE_ENV` | No | `development` | Environment mode: `development`, `production`, `test` |
| `PORT` | No | `3001` | API server port |
| `API_BASE_URL` | No | `http://localhost:3001` | Public API URL for callbacks |
| `INSTANCE_ID` | Yes | - | Unique platform instance identifier |
| `CORS_ORIGINS` | Yes | - | Comma-separated allowed origins |

---

## Database

### MongoDB

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | Yes | - | MongoDB connection string |
| `MONGODB_REPLICA_SET` | No | - | Replica set name for HA |

**Examples:**
```bash
# Local
MONGODB_URI=mongodb://localhost:27017/relay_one

# Atlas
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/relay_one

# Replica Set
MONGODB_URI=mongodb://host1:27017,host2:27017/relay_one?replicaSet=rs0
```

### Redis

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | No | - | Redis connection URL |
| `REDIS_SENTINEL_URLS` | No | - | Redis Sentinel URLs for HA |

**Examples:**
```bash
# Single instance
REDIS_URL=redis://localhost:6379

# With password
REDIS_URL=redis://:password@localhost:6379

# Sentinel
REDIS_SENTINEL_URLS=redis://sentinel1:26379,redis://sentinel2:26379
```

### ClickHouse (Analytics)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLICKHOUSE_HOST` | No | `localhost` | ClickHouse host |
| `CLICKHOUSE_PORT` | No | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DB` | No | `relay_one` | Database name |
| `CLICKHOUSE_USER` | No | `relay` | Username |
| `CLICKHOUSE_PASSWORD` | No | `relay_secret` | Password |
| `CLICKHOUSE_SECURE` | No | `false` | Use HTTPS |

---

## Authentication

### JWT

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET` | Yes | - | Secret for signing JWTs (min 32 chars) |
| `JWT_ISSUER` | No | `relay.one` | JWT issuer claim |
| `JWT_EXPIRY` | No | `24h` | Token expiry duration |

### Encryption

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENCRYPTION_KEY` | Prod | - | 32-byte key for data encryption |
| `WORKFLOW_ENCRYPTION_KEY` | Prod | - | Hex key for workflow credentials |

**Generate Keys:**
```bash
# JWT Secret
openssl rand -base64 48

# Encryption Key (32 bytes hex)
openssl rand -hex 32
```

---

## SSO / Identity

### SAML

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SAML_ENTITY_ID` | No | Auto | SAML Service Provider entity ID |
| `SAML_ACS_URL` | No | Auto | Assertion Consumer Service URL |

### OIDC

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OIDC_CLIENT_ID` | No | - | OAuth client ID |
| `OIDC_CLIENT_SECRET` | No | - | OAuth client secret |
| `OIDC_ISSUER` | No | - | OIDC issuer URL |

---

## Peering & Federation

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PEERING_ENDPOINT` | No | - | External URL for inbound peering |
| `PEER_INSTANCES` | No | - | Comma-separated peer API URLs |

---

## Central Registry

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CENTRAL_REGISTRY_URL` | No | - | Central registry sync URL |
| `CENTRAL_REGISTRY_API_KEY` | Cond | - | API key (required if URL set) |
| `DEPLOYMENT_ID` | No | - | Unique deployment identifier |

---

## RelayChain Blockchain

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RELAYCHAIN_RPC_URL` | No | `http://localhost:8545` | RPC endpoint |
| `RELAYCHAIN_CHAIN_ID` | No | `relay-mainnet` | Chain identifier |
| `RELAYCHAIN_PRIVATE_KEY` | No | - | Platform wallet private key |

---

## Certificates & mTLS

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RELAY_CA_URL` | No | `https://ca.relay.one` | Central CA URL |
| `RELAY_CA_API_KEY` | No | - | CA API key |
| `RELAY_CA_ENABLED` | No | `true` | Enable central CA |
| `RELAY_ORGANIZATION_ID` | No | `default` | Organization for certs |
| `ROOT_CA_CERTIFICATE` | No | - | Root CA PEM certificate |
| `ROOT_CA_PRIVATE_KEY` | No | - | Root CA private key |

---

## Payments & Billing

### Stripe

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STRIPE_SECRET_KEY` | No | - | Stripe API secret key |
| `STRIPE_WEBHOOK_SECRET` | No | - | Stripe webhook signing secret |
| `STRIPE_PUBLISHABLE_KEY` | No | - | Stripe publishable key (frontend) |

---

## LLM Providers

### OpenAI

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | OpenAI API key |
| `OPENAI_ORG_ID` | No | - | OpenAI organization ID |

### Anthropic

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |

### Google AI

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | No | - | Google AI API key |
| `VERTEX_AI_PROJECT_ID` | No | - | Vertex AI project |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | No | - | Service account JSON |

### Local Models (Ollama)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama server URL |

---

## Integrations

### Slack

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SLACK_CLIENT_ID` | No | - | Slack OAuth client ID |
| `SLACK_CLIENT_SECRET` | No | - | Slack OAuth client secret |
| `SLACK_SIGNING_SECRET` | No | - | Slack request signing secret |
| `SLACK_BOT_TOKEN` | No | - | Slack bot token |
| `SLACK_ENCRYPTION_KEY` | No | - | Key for encrypting Slack tokens |
| `SLACK_WEBHOOK_URL` | No | - | Slack webhook for alerts |

### GitHub

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_TOKEN` | No | - | GitHub personal access token |
| `GITHUB_APP_ID` | No | - | GitHub App ID |
| `GITHUB_PRIVATE_KEY` | No | - | GitHub App private key |

### Glean

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GLEAN_API_KEY` | No | - | Glean API key |
| `GLEAN_INSTANCE_URL` | No | - | Glean instance URL |

---

## Email

### SendGrid

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SENDGRID_API_KEY` | No | - | SendGrid API key |

### Resend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RESEND_API_KEY` | No | - | Resend API key |

### AWS SES

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_SES_REGION` | No | `us-east-1` | SES region |
| `ALERT_FROM_EMAIL` | No | `alerts@relay.one` | From address |

---

## Cloud Storage

### AWS S3

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | Cond | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Cond | - | AWS secret key |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `REPORT_S3_BUCKET` | No | `relay-one-reports` | Reports bucket |
| `EXPORT_S3_BUCKET` | No | `relay-one-exports` | Exports bucket |

### Google Cloud Storage

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_CLIENT_EMAIL` | Cond | - | Service account email |
| `GCP_PRIVATE_KEY` | Cond | - | Service account key |
| `REPORT_GCS_BUCKET` | No | `relay-one-reports` | Reports bucket |
| `EXPORT_GCS_BUCKET` | No | `relay-one-exports` | Exports bucket |

### Azure Blob Storage

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_STORAGE_ACCOUNT` | Cond | - | Storage account name |
| `AZURE_STORAGE_KEY` | Cond | - | Storage account key |
| `AZURE_STORAGE_CONTAINER` | No | `reports` | Container name |
| `EXPORT_AZURE_CONTAINER` | No | `exports` | Exports container |

---

## Alerting

### PagerDuty

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PAGERDUTY_ROUTING_KEY` | No | - | PagerDuty routing key |

### OpsGenie

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPSGENIE_API_KEY` | No | - | OpsGenie API key |

---

## Monitoring & Observability

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTEL_ENABLED` | No | `false` | Enable OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | - | OTLP collector endpoint |
| `OTEL_SERVICE_NAME` | No | `relay-api` | Service name in traces |
| `OTEL_SAMPLING_RATE` | No | `1.0` | Trace sampling rate (0-1) |
| `LOG_FORMAT` | No | `json` | Log format: `json` or `text` |
| `LOG_LEVEL` | No | `info` | Log level: `debug`, `info`, `warn`, `error` |

---

## Rate Limiting

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RATE_LIMIT_WINDOW` | No | `60000` | Window in milliseconds |
| `RATE_LIMIT_MAX_TOKENS` | No | `100` | Max requests per window |

---

## Security

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QUANTUM_SAFE_MODE` | No | `false` | Enable post-quantum crypto |
| `CIPHER_ALGORITHM` | No | `aes-256-gcm` | Encryption algorithm |
| `KEY_ROTATION_DAYS` | No | `90` | Key rotation interval |
| `CERT_AUTO_RENEW` | No | `true` | Auto-renew certificates |

---

## Post-Quantum Cryptography

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QUANTUM_PROVIDER` | No | `auto` | PQC provider: `liboqs`, `simulated`, `auto` |
| `QUANTUM_ALLOW_SIMULATED` | No | `false` | Allow simulated PQC (dev/test only) |
| `LIBOQS_KEM_DEFAULT` | No | `ML-KEM-768` | Default KEM algorithm |
| `LIBOQS_DSA_DEFAULT` | No | `ML-DSA-65` | Default signature algorithm |

**Notes:**
- `auto` mode tries liboqs first, falls back to simulated if `QUANTUM_ALLOW_SIMULATED=true`
- **NEVER** set `QUANTUM_ALLOW_SIMULATED=true` in production
- Requires `liboqs-node` npm package and liboqs C library for real PQC

---

## Swarm Coordinator

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SWARM_SIMULATED_EXECUTION` | No | `false` | Use simulated task execution (dev only) |
| `SWARM_DEFAULT_TIMEOUT_MS` | No | `300000` | Default task timeout (5 minutes) |
| `SWARM_MAX_WORKERS_PER_SWARM` | No | `50` | Maximum workers per swarm |

**Notes:**
- When `SWARM_SIMULATED_EXECUTION=true`, tasks use random delays instead of real agents
- **NEVER** set this in production - tasks won't actually execute

---

## Vulnerability Database

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVD_API_KEY` | Recommended | - | NVD API key for higher rate limits |
| `CVE_CACHE_TTL_SECONDS` | No | `3600` | CVE cache TTL (1 hour default) |
| `CVE_ENABLE_NVD` | No | `true` | Enable NVD integration |
| `CVE_ENABLE_OSV` | No | `true` | Enable OSV integration |
| `CVE_RATE_LIMIT_PER_MIN` | No | `30` | API rate limit per minute |

**Notes:**
- Get an NVD API key from https://nvd.nist.gov/developers/request-an-api-key
- Without an API key, NVD rate limits to 5 requests per 30 seconds
- With an API key, NVD allows 50 requests per 30 seconds

---

## Feature Flags

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEATURE_QUANTUM_SAFE` | No | `false` | Enable quantum-safe features |
| `FEATURE_WORKFLOWS` | No | `true` | Enable workflow engine |
| `FEATURE_RELAYCHAIN` | No | `true` | Enable blockchain features |
| `FEATURE_FEDERATION` | No | `true` | Enable federation |

---

## Storage Paths

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REPORT_STORAGE_TYPE` | No | `local` | Storage: `local`, `s3`, `gcs`, `azure` |
| `EXPORT_STORAGE_TYPE` | No | `local` | Storage type for exports |
| `EXPORT_LOCAL_DIR` | No | `/tmp/exports` | Local export directory |

---

## Webhooks

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CAUSALITY_WEBHOOK_SECRET` | No | - | Secret for signing webhooks |
| `REMEDIATION_WORKFLOW_URL` | No | - | URL for auto-remediation |

---

## Agent Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PLATFORM_API_URL` | No | - | Platform URL for agents |
| `AGENT_HOST` | No | `localhost` | Agent advertised host |

---

## Frontend Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_URL` | No | - | API URL for browser |
| `NEXT_PUBLIC_GATEWAY_URL` | No | - | Gateway URL (Next.js) |
| `NEXT_PUBLIC_API_URL` | No | - | API URL (Next.js) |

---

## Example Configurations

### Local Development

```bash
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/relay_one
JWT_SECRET=dev-secret-at-least-32-characters-long
INSTANCE_ID=local-dev
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002
API_URL=http://localhost:3001
OLLAMA_URL=http://localhost:11434
```

### Production (Minimal)

```bash
NODE_ENV=production
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/relay_one
JWT_SECRET=$(openssl rand -base64 48)
ENCRYPTION_KEY=$(openssl rand -hex 32)
INSTANCE_ID=prod-1
CORS_ORIGINS=https://console.relay.one,https://admin.relay.one
API_BASE_URL=https://api.relay.one
STRIPE_SECRET_KEY=sk_live_xxx
```

### Production (Full)

```bash
# Core
NODE_ENV=production
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/relay_one
REDIS_URL=redis://:password@redis.internal:6379
JWT_SECRET=$(openssl rand -base64 48)
ENCRYPTION_KEY=$(openssl rand -hex 32)
WORKFLOW_ENCRYPTION_KEY=$(openssl rand -hex 32)
INSTANCE_ID=prod-us-east-1
API_BASE_URL=https://api.relay.one
CORS_ORIGINS=https://console.relay.one,https://admin.relay.one

# Analytics
CLICKHOUSE_HOST=clickhouse.internal
CLICKHOUSE_PASSWORD=secure-password
CLICKHOUSE_SECURE=true

# Payments
STRIPE_SECRET_KEY=sk_live_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx

# LLM
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx

# Integrations
SLACK_CLIENT_ID=xxx
SLACK_CLIENT_SECRET=xxx
SLACK_SIGNING_SECRET=xxx
GITHUB_TOKEN=ghp_xxx

# Storage
REPORT_STORAGE_TYPE=s3
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1

# Monitoring
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.internal:4317
LOG_FORMAT=json
LOG_LEVEL=info

# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
PAGERDUTY_ROUTING_KEY=xxx

# Security
QUANTUM_SAFE_MODE=false
KEY_ROTATION_DAYS=90
```

---

## See Also

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment instructions
- [SECURITY.md](SECURITY.md) - Security configuration
- [docs/guides/integrations.md](docs/guides/integrations.md) - Integration setup
