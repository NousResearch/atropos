# relay.one Deployment Guide

Comprehensive guide for deploying relay.one in development, staging, and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Deployment Options](#deployment-options)
4. [Docker Compose Deployment](#docker-compose-deployment)
5. [Kubernetes Deployment (Helm)](#kubernetes-deployment-helm)
6. [DigitalOcean App Platform](#digitalocean-app-platform)
7. [Security Configuration](#security-configuration)
8. [Database Setup](#database-setup)
9. [Monitoring Setup](#monitoring-setup)
10. [Demo Agents](#demo-agents)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| Memory | 8 GB | 16+ GB |
| Storage | 50 GB SSD | 200+ GB SSD |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

| Software | Version | Required For |
|----------|---------|--------------|
| Docker | 24.0+ | All deployments |
| Docker Compose | 2.20+ | Docker deployment |
| Kubernetes | 1.28+ | K8s deployment |
| Helm | 3.12+ | K8s deployment |
| Node.js | 20.x | Development |
| pnpm | 8.x | Development |
| Rust | 1.75+ | Gateway compilation |

### External Dependencies

| Service | Required | Purpose |
|---------|----------|---------|
| MongoDB | Yes | Primary database (6.3+) |
| Redis | Yes | Caching, sessions, pub/sub (7.0+) |
| ClickHouse | Optional | Analytics (high-volume) |
| Stripe | Optional | Billing integration |
| SMTP Provider | Optional | Email notifications |

---

## Quick Start

### Development Mode

```bash
# Prerequisites: Node.js 20+, pnpm 8+, Docker

# 1. Start MongoDB
docker run -d -p 27017:27017 --name relay-mongo mongo:7

# 2. Install dependencies
pnpm install

# 3. Seed demo data
cd scripts && npx tsx seed-db.ts && cd ..

# 4. Start all services
pnpm dev

# Or start individually:
# pnpm dev:api      - API Server (:3001)
# pnpm dev:console  - Console Portal (:3000)
# pnpm dev:admin    - Admin Portal (:3002)
```

**Demo Credentials:** `demo@relay.one` / `demo123`

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Console | http://localhost:3000 | Customer dashboard |
| API Server | http://localhost:3001 | Core API |
| Admin Portal | http://localhost:3002 | Admin dashboard |
| Gateway Rust | http://localhost:3100 | High-performance gateway |
| API Docs (Swagger) | http://localhost:3001/docs | Interactive API docs |
| Weather Agent | http://localhost:5001 | Demo agent |
| Data Analyst | http://localhost:5002 | Demo agent |
| Risky Agent | http://localhost:5003 | Demo agent |

---

## Deployment Options

| Option | Best For | Complexity | Scalability |
|--------|----------|------------|-------------|
| **Docker Compose** | Single node, dev/staging | Low | Limited |
| **Kubernetes (Helm)** | Production, multi-node | Medium | High |
| **DigitalOcean** | Quick cloud deployment | Low | Medium |
| **Manual** | Custom infrastructure | High | Custom |

---

## Docker Compose Deployment

### Quick Deploy

```bash
# Navigate to deploy directory
cd deploy/docker

# Copy and configure environment
cp .env.example .env
# Edit .env with your secrets (see Security Configuration)

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Docker Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Production stack |
| `docker-compose.dev.yml` | Development with hot reload |
| `docker-compose.platform.yml` | Full platform stack |
| `docker-compose.agents.yml` | Demo agents only |

### Deploy Script

```bash
cd deploy

# Deploy full platform (API + Console + Gateway Rust + MongoDB + Redis)
./deploy.sh platform

# Or deploy everything including demo agents
./deploy.sh all

# Check status
./deploy.sh status

# View logs
./deploy.sh logs gateway-rust

# Stop all services
./deploy.sh stop
```

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| console | 3000 | Customer dashboard |
| api | 3001 | Node.js API server |
| admin | 3002 | Admin portal |
| gateway-rust | 3100 | High-performance Rust gateway |
| mongodb | 27017 | Primary database |
| redis | 6379 | Cache and rate limiting |

---

## Kubernetes Deployment (Helm)

### Quick Start

```bash
# Create namespace
kubectl create namespace relay-one

# Create secrets first (see Security Configuration)
kubectl create secret generic relay-one-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --from-literal=encryption-key=$(openssl rand -hex 32) \
  --from-literal=mongodb-uri="mongodb://..." \
  -n relay-one

# Install from local chart
helm install relay-one ./deploy/kubernetes/helm/relay-one-appliance \
  --namespace relay-one

# Or with custom values
helm install relay-one ./deploy/kubernetes/helm/relay-one-appliance \
  --namespace relay-one \
  --set api.replicas=3 \
  --set gatewayRust.replicas=5 \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=relay.example.com
```

### Custom Values Configuration

```yaml
# custom-values.yaml
api:
  replicas: 3
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

gatewayRust:
  replicas: 5
  resources:
    requests:
      cpu: 1000m
      memory: 256Mi
    limits:
      cpu: 4000m
      memory: 512Mi

console:
  replicas: 2

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
  tls:
    - secretName: relay-one-tls
      hosts:
        - relay.example.com
        - api.relay.example.com

mongodb:
  external:
    enabled: true
    uri: "mongodb+srv://..."

redis:
  external:
    enabled: true
    url: "redis://..."

monitoring:
  serviceMonitor:
    enabled: true
```

### Helm Operations

```bash
# Check status
kubectl get pods -n relay-one

# View gateway-rust logs
kubectl logs -f deployment/relay-one-gateway-rust -n relay-one

# Upgrade deployment
helm upgrade relay-one ./deploy/kubernetes/helm/relay-one-appliance \
  --namespace relay-one \
  --values custom-values.yaml

# Rollback to previous version
helm rollback relay-one 1 -n relay-one

# Uninstall
helm uninstall relay-one -n relay-one
```

---

## DigitalOcean App Platform

```bash
# Install doctl CLI
brew install doctl  # macOS

# Authenticate
doctl auth init

# Create app from spec
doctl apps create --spec deploy/digitalocean/app-spec.yaml

# Update app
doctl apps update <app-id> --spec deploy/digitalocean/app-spec.yaml
```

---

## Security Configuration

### Generating Secure Secrets

```bash
# JWT Secret (64 bytes, base64 encoded)
openssl rand -base64 64

# Encryption Key (32 bytes, hex encoded - exactly 64 hex chars)
openssl rand -hex 32

# Alternative: Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### Required Secrets

| Secret | Format | Length | Description |
|--------|--------|--------|-------------|
| `JWT_SECRET` | Base64 | 64+ chars | JWT signing secret |
| `ENCRYPTION_KEY` | Hex | 64 chars (32 bytes) | Data encryption key |
| `MONGODB_URI` | URI | - | MongoDB connection string |

### Environment File Template

```bash
# .env.production
NODE_ENV=production

# Core secrets (REQUIRED - generate securely!)
JWT_SECRET=<generate-with-openssl>
ENCRYPTION_KEY=<generate-with-openssl>

# Database
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/relay_one
REDIS_URL=redis://:password@redis-host:6379

# API Configuration
API_PORT=3001
GATEWAY_PORT=3100
CORS_ORIGINS=https://console.relay.example.com,https://admin.relay.example.com

# Optional: Stripe Billing
STRIPE_SECRET_KEY=sk_live_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx

# Optional: Email
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=SG.xxx
```

### Gateway Rust Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `GATEWAY_PORT` | `3100` | Listen port |
| `GATEWAY_WORKERS` | `0` | Worker threads (0 = auto) |
| `RUST_LOG` | `info` | Log level |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS_PER_SECOND` | `1000` | Requests per second limit |
| `GOVERNANCE_PII_DETECTION_ENABLED` | `true` | Enable PII detection |
| `BILLING_ENABLED` | `true` | Enable billing checks |

### Security Checklist

- [ ] Strong JWT_SECRET (32+ characters, cryptographically random)
- [ ] Strong ENCRYPTION_KEY (exactly 32 bytes / 64 hex chars)
- [ ] HTTPS enabled (use ingress TLS or load balancer)
- [ ] MongoDB authentication enabled
- [ ] Redis password configured
- [ ] Rate limiting configured
- [ ] CORS configured for your domains
- [ ] Network policies in Kubernetes
- [ ] Secrets stored in external secret manager (Vault, AWS, etc.)

---

## Database Setup

### MongoDB

```bash
# Docker (development)
docker run -d -p 27017:27017 --name relay-mongo mongo:7

# Production replica set
rs.initiate({
  _id: "relay-rs",
  members: [
    { _id: 0, host: "mongo1:27017", priority: 2 },
    { _id: 1, host: "mongo2:27017", priority: 1 },
    { _id: 2, host: "mongo3:27017", priority: 1 }
  ]
});
```

**Connection String Options:**
```
mongodb+srv://user:pass@cluster.mongodb.net/relay_one?retryWrites=true&w=majority&readPreference=primaryPreferred
```

### Redis

```bash
# Docker (development)
docker run -d -p 6379:6379 --name relay-redis redis:7

# Production with persistence
redis-server --appendonly yes --requirepass "secure-password"
```

---

## Monitoring Setup

### Health Checks

```bash
# API health
curl http://localhost:3001/health

# Gateway health
curl http://localhost:3100/health

# Gateway readiness
curl http://localhost:3100/ready

# Prometheus metrics
curl http://localhost:3100/metrics
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'relay-one-api'
    static_configs:
      - targets: ['api:3001']
    metrics_path: /metrics

  - job_name: 'relay-one-gateway'
    static_configs:
      - targets: ['gateway-rust:3100']
    metrics_path: /metrics
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request latency |
| `gateway_invocations_total` | Counter | Agent invocations |
| `gateway_active_connections` | Gauge | Current connections |
| `hitl_pending_approvals` | Gauge | Pending HITL requests |

---

## Demo Agents

Three demo agents are included in `apps/demo-agents/`:

| Agent | Port | Trust Level | Description |
|-------|------|-------------|-------------|
| weather-agent | 5001 | HIGH | Unrestricted weather data |
| data-analyst | 5002 | MEDIUM | Rate limited, PII detection |
| risky-agent | 5003 | LOW | All operations require HITL approval |

### Starting Demo Agents

```bash
# Start all agents
cd apps/demo-agents && pnpm dev

# Or start individually
pnpm --filter @relay-one/demo-agents start:weather
pnpm --filter @relay-one/demo-agents start:analyst
pnpm --filter @relay-one/demo-agents start:risky
```

### Testing Agent Invocation

```bash
# Via Rust Gateway (recommended for high-throughput)
curl -X POST http://localhost:3100/gateway/demo/agents/weather-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_weather", "parameters": {"location": "Seattle"}}'

# Via Node.js API
curl -X POST http://localhost:3001/gateway/demo/agents/weather-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_weather", "parameters": {"location": "Seattle"}}'
```

### Demo Scenarios

1. **Agent Invocation**: Call weather agent through the gateway
2. **PII Blocking**: Send SSN (`123-45-6789`) to see governance block it
3. **HITL Approval**: Invoke risky-agent (requires human approval)
4. **Rate Limiting**: Rapidly invoke data-analyst to trigger limits

---

## Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
docker ps | grep mongo

# Check MongoDB logs
docker logs relay-mongo

# Restart MongoDB
docker restart relay-mongo
```

### API Not Starting

```bash
# Check port is available
lsof -i :3001

# Verify environment
cat .env.local

# Check for TypeScript errors
pnpm typecheck
```

### Agents Not Connecting

1. Ensure API is running on port 3001
2. Check agent logs for registration errors
3. Verify MongoDB has demo data seeded

### Port Conflicts

```bash
# Find process using port
lsof -i :3001

# Kill if needed
kill -9 <PID>
```

### Viewing Logs

```bash
# Docker logs
docker-compose logs -f api
docker-compose logs -f gateway-rust

# Kubernetes logs
kubectl logs -f deployment/api -n relay-one
kubectl logs -f deployment/gateway-rust -n relay-one
```

---

## Performance Benchmarks

The Rust gateway provides significant performance improvements over Node.js:

| Metric | Node.js Gateway | Rust Gateway | Improvement |
|--------|-----------------|--------------|-------------|
| Requests/sec | ~10,000 | ~100,000+ | **10x** |
| P50 Latency | ~15ms | ~1-2ms | **7-10x** |
| P99 Latency | ~100ms | ~10ms | **10x** |
| Memory Usage | ~500MB | ~50MB | **10x** |
| CPU at 10k RPS | 100% | 20% | **5x** |

---

## Production Checklist

### Pre-Deployment

- [ ] Generate cryptographically secure secrets
- [ ] Configure MongoDB with authentication and TLS
- [ ] Configure Redis with password
- [ ] Set up TLS certificates
- [ ] Configure CORS origins
- [ ] Set up monitoring endpoints
- [ ] Configure backup schedule
- [ ] Review rate limiting settings

### Post-Deployment

- [ ] Verify all services healthy
- [ ] Test authentication flow
- [ ] Test agent creation and invocation
- [ ] Verify monitoring data flowing
- [ ] Test backup and restore
- [ ] Set up alerting rules
- [ ] Document runbook

---

## Support

- **Documentation**: https://docs.relay.one
- **Issues**: https://github.com/relay-one/relay-one/issues
- **Email**: support@relay.one

---

*Last updated: December 2025*
