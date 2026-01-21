# relay.one

Enterprise AI Agent Infrastructure Platform - Production-ready gateway for secure, governed agentic workflows.

[![License](https://img.shields.io/badge/license-proprietary-blue.svg)]()
[![Node.js](https://img.shields.io/badge/node-20.x-green.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)]()

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          relay.one Architecture                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                    relay.one Central (SaaS)                         │    │
│   │                 https://relay.one - Vercel Hosted                   │    │
│   │                                                                     │    │
│   │   • Agent Registry & Discovery    • Certificate Authority          │    │
│   │   • Reputation System             • External Developer Portal      │    │
│   │   • Discovery & Reputation MCP Servers                             │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                              Optional Sync                                   │
│                                    │                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │              Enterprise Appliances (On-Prem/Private Cloud)          │    │
│   │              Docker Swarm / Kubernetes / Private Cloud              │    │
│   │                                                                     │    │
│   │   • Agent Gateway (Rust: 100k+ RPS)  • Human-in-the-Loop Approvals │    │
│   │   • Governance & Policy Engine       • Certificate Management       │    │
│   │   • Billing & Usage Tracking         • Complete Data Sovereignty    │    │
│   │   • RelayChain Blockchain            • A2A Protocol                 │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Documentation

| Document | Description |
|----------|-------------|
| **[Quick Start](#quick-start)** | Get running in minutes |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design and components |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | Production deployment |
| **[ENVIRONMENT.md](ENVIRONMENT.md)** | All environment variables |
| **[SECURITY.md](SECURITY.md)** | Security configuration |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Development guide |
| **[docs/](docs/README.md)** | Full documentation

### API & SDK Documentation

| Document | Description |
|----------|-------------|
| [docs/api/](docs/api/README.md) | API reference |
| [packages/sdk/](packages/sdk/README.md) | TypeScript SDK |
| [packages/sdk-python/](packages/sdk-python/README.md) | Python SDK |
| [packages/sdk-rust/](packages/sdk-rust/README.md) | Rust SDK |
| [packages/sdk-go/](packages/sdk-go/README.md) | Go SDK |

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/relay-one/relay-one.git
cd relay-one/deploy/docker
cp .env.example .env
# Edit .env with your secrets (see SECURITY.md)
docker-compose up -d
```

**Access:**
- Console: http://localhost:3000
- API: http://localhost:3001
- Gateway: http://localhost:3100
- API Docs: http://localhost:3001/docs

### Option 2: Development Mode

```bash
# Prerequisites: Node.js 20+, pnpm 8+, Docker

# Install and start
pnpm install
docker run -d -p 27017:27017 --name relay-mongo mongo:7
cd scripts && npx tsx seed-db.ts && cd ..
pnpm dev
```

**Demo Credentials:** `demo@relay.one` / `demo123`

### Option 3: Kubernetes (Helm)

```bash
helm install relay-one ./deploy/kubernetes/helm/relay-one-appliance \
  --namespace relay-one --create-namespace \
  --set secrets.jwtSecret=$(openssl rand -base64 32) \
  --set secrets.encryptionKey=$(openssl rand -hex 32)
```

---

## Platform Overview

relay.one provides enterprise AI agent governance with two deployment models:

| Model | Description |
|-------|-------------|
| **Appliance** | Self-hosted for complete data sovereignty |
| **Central** | SaaS for discovery, reputation, certificates |

### Key Features

| Feature | Description |
|---------|-------------|
| **Agent Gateway** | Rust gateway handling 100k+ RPS |
| **Governance Engine** | PII detection, rate limiting, content filtering |
| **HITL Approvals** | Human approval for sensitive operations |
| **A2A Protocol** | Cryptographically signed agent messaging |
| **RelayChain** | Blockchain for payments and identity |
| **Multi-Protocol** | MCP, REST, WebSocket, gRPC support |
| **Quantum-Safe Crypto** | NIST FIPS 203/204/205/206 compliant PQC |
| **19 Capability Types** | Text, image, audio, video, code, data analysis |
| **TEE Security** | Intel SGX, AMD SEV, ARM TrustZone |
| **Multi-Cloud** | Docker, K8s, GKE, EKS, DigitalOcean |

---

## Architecture

### Applications

| App | Port | Description |
|-----|------|-------------|
| **api** | 3001 | Fastify API (71 routes, 124 services) |
| **gateway-rust** | 3100 | High-performance Rust gateway |
| **console** | 3000 | Customer dashboard (61 pages) |
| **admin** | 3002 | Admin portal (25 pages) |
| **mcp** | - | MCP protocol server |
| **relay-one-web** | - | Marketing site & Central portal (18 pages, deep-dive features) |
| **relay-one-api** | - | Central registry API |
| **discovery-mcp** | - | Discovery MCP server |
| **reputation-mcp** | - | Reputation MCP server |
| **demo-agents** | 5001-5003 | Demo agents |

### Packages

| Package | Description |
|---------|-------------|
| **@relay-one/sdk** | TypeScript SDK (14 modules) |
| **@relay-one/types** | Shared TypeScript types |
| **@relay-one/database** | MongoDB client |
| **@relay-one/ui** | React UI components |
| **sdk-python** | Python SDK |
| **sdk-rust** | Rust SDK |
| **sdk-go** | Go SDK |

### RelayChain Blockchain (28 Rust Crates)

Production-ready blockchain infrastructure with comprehensive security and high availability:

| Category | Crates | Features |
|----------|--------|----------|
| **Core** | relay-chain-core, relay-chain-node | Block production, state management, networking |
| **Consensus** | relay-chain-consensus, relay-chain-quorum | Raft consensus, multi-signature quorum |
| **Security** | relay-chain-integrity, relay-chain-tee | Ed25519 signatures, SGX/SEV TEE support |
| **Networking** | relay-chain-network, relay-chain-gateway | QUIC P2P, priority ordering queue |
| **Identity** | relay-chain-ownership, relay-chain-identity | Per-chain wallet ownership, fencing tokens |
| **Payments** | relay-chain-payments, relay-chain-compliance | Transaction processing, regulatory compliance |
| **Infrastructure** | relay-chain-backup, relay-chain-loadbalancer | AES-256-GCM encrypted backups, 7 LB strategies |
| **MetaMask** | relay-chain-eth-rpc, relay-chain-bridge | Full Ethereum JSON-RPC, EIP-1559 support |

**Key Capabilities:**
- **Multi-Node Orchestration**: Run 10+ nodes on a single host with automatic port allocation
- **Split-Brain Prevention**: Fencing tokens, lease expiration, heartbeat detection
- **TEE Support**: Intel SGX, AMD SEV, ARM TrustZone (simulation mode for development)
- **Recovery**: GenesisRecoveryKey enforcement, GENESIS_AUDIT_HASH permanent markers
- **Load Balancing**: Round-Robin, Least Connections, IP Hash, Weighted, Random, Least Response Time, Resource-Aware

---

## Console Features

The customer console provides:

| Section | Features |
|---------|----------|
| **Dashboard** | Live stats, pending approvals, agent list, activity feed |
| **Agents** | Create, configure, monitor agents |
| **Governance** | PII detection, policies, audit logs |
| **Approvals** | HITL approval queue with approve/reject/escalate |
| **RelayChain** | Faucet, network status, agent wallets |
| **Billing** | Plans, invoices, usage tracking |
| **Analytics** | Real-time metrics, events, alerts |

---

## Admin Features

Platform-wide administration:

| Section | Features |
|---------|----------|
| **Users** | Manage all users, roles, MFA status |
| **Organizations** | Manage orgs, plans, limits |
| **Agents** | Platform-wide agent control |
| **Governance** | Global policies, spending limits |
| **Certificates** | mTLS certificate approval |
| **Analytics** | Revenue, usage, performance |
| **Violations** | Policy violation tracking |
| **Network Health** | Service status monitoring |

---

## API Quick Reference

| Category | Endpoints |
|----------|-----------|
| Auth | `/api/v1/auth/*` |
| Agents | `/api/v1/agents/*` |
| Gateway | `/gateway/*` |
| Governance | `/governance/*` |
| HITL | `/hitl/*` |
| Billing | `/billing/*` |
| Certificates | `/api/v1/certificates/*` |
| Discovery | `/api/v1/discovery/*` |

**Full API documentation:** [docs/api/README.md](docs/api/README.md)

---

## SDK Usage

```typescript
import { RelayClient } from '@relay-one/sdk';

const client = new RelayClient({
  apiKey: 'rly_your_api_key',
  baseUrl: 'http://localhost:3001',
});

// Invoke agent
const result = await client.gatewayInvoke('weather-agent', {
  tool: 'get_weather',
  input: { location: 'Seattle' },
});
```

---

## MCP Integration

### Discovery MCP

```json
{
  "mcpServers": {
    "relay-discovery": {
      "command": "npx",
      "args": ["relay-one-discovery-mcp"],
      "env": { "RELAY_ONE_API_KEY": "your-key" }
    }
  }
}
```

**Tools:** `search_agents`, `get_agent`, `invoke_agent`, `list_capabilities`

### Reputation MCP

**Tools:** `get_agent_reputation`, `get_reputation_history`, `get_leaderboard`, `compare_agents`

---

## Development

### Scripts

| Script | Description |
|--------|-------------|
| `pnpm dev` | Start all apps |
| `pnpm build` | Build for production |
| `pnpm test` | Run tests |
| `pnpm lint` | Lint code |

### Demo Agents

| Agent | Port | Trust | Description |
|-------|------|-------|-------------|
| weather-agent | 5001 | HIGH | Weather data |
| data-analyst | 5002 | MEDIUM | Rate limited |
| risky-agent | 5003 | LOW | HITL required |

---

## Project Structure

```
relay.one/
├── apps/
│   ├── api/                 # Fastify API (71 routes, 121 services)
│   ├── gateway-rust/        # Rust gateway
│   ├── console/             # Customer portal
│   ├── admin/               # Admin portal
│   ├── relay-one-web/       # Marketing site
│   ├── relay-one-api/       # Central API
│   ├── discovery-mcp/       # Discovery MCP
│   ├── reputation-mcp/      # Reputation MCP
│   ├── mcp/                 # Appliance MCP
│   └── demo-agents/         # Demo agents
├── packages/
│   ├── sdk/                 # TypeScript SDK
│   ├── sdk-python/          # Python SDK
│   ├── sdk-rust/            # Rust SDK
│   ├── sdk-go/              # Go SDK
│   ├── types/               # Shared types
│   ├── database/            # MongoDB client
│   ├── ui/                  # UI components
│   └── config/              # Shared configs
├── relay-chain/             # Blockchain (28 crates)
├── docs/                    # Documentation
├── tests/                   # Test suites
└── deploy/                  # Deployment configs
```

---

## Configuration

### Required Variables

| Variable | Description |
|----------|-------------|
| `JWT_SECRET` | JWT signing secret (32+ chars) |
| `ENCRYPTION_KEY` | Encryption key (32 bytes hex) |
| `MONGODB_URI` | MongoDB connection |

See [SECURITY.md](SECURITY.md) for secure generation.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Next.js 14, React 18, Tailwind, Radix UI |
| **Backend** | Fastify 4.24, Axum (Rust), TypeScript |
| **Database** | MongoDB 6.3+, Redis 7.0, ClickHouse |
| **Auth** | JWT, mTLS, Clerk |
| **Payments** | Stripe, X.402, AP2 |
| **Monitoring** | Prometheus, Grafana, OpenTelemetry |

---

## Scale Metrics

| Metric | Value |
|--------|-------|
| Applications | 12 |
| API Routes | 71 |
| API Services | 124 |
| Console Pages | 61 |
| Admin Pages | 25 |
| SDK Modules | 13 |
| Rust Crates | 28 |
| Test Files | 60+ |

---

## Support

- **Documentation:** [docs/](docs/README.md)
- **Issues:** https://github.com/relay-one/relay-one/issues
- **Email:** support@relay.one

---

## License

Copyright 2024-2025 relay.one. All rights reserved.
