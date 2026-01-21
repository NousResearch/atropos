# relay.one - Investor Quick Start

> **Enterprise AI Agent Governance Gateway** - Production-ready platform

## 60-Second Start

```bash
# Prerequisites: Docker, Node.js 20+, pnpm 8+

# 1. Start MongoDB
docker run -d -p 27017:27017 --name relay-mongo mongo:7

# 2. Install and seed
pnpm install
cd scripts && npx tsx seed-db.ts && cd ..

# 3. Start API and Demo Agents
pnpm dev:api                      # Terminal 1: API on :3001
cd apps/demo-agents && pnpm dev   # Terminal 2: Agents on :4001-4003
```

## Access Points

| Service | Port | Description |
|---------|------|-------------|
| **API Gateway** | 3001 | Core relay - all agent traffic flows here |
| **Console** | 3000 | Customer dashboard |
| **Admin** | 3002 | Platform operator dashboard |
| **Gateway Rust** | 3100 | High-performance gateway (100k+ RPS) |
| **Swagger Docs** | 3001/docs | Interactive API documentation |

## Demo Credentials

- **Email:** `demo@relay.one`
- **Password:** `demo123`

## Demo Agents

| Agent | Port | Trust | Description |
|-------|------|-------|-------------|
| weather-agent | 4001 | HIGH | Unrestricted weather data |
| data-analyst | 4002 | MEDIUM | Rate limited, PII detection |
| risky-agent | 4003 | LOW | Requires HITL approval |

---

## What to Demo

### 1. Real Agent Invocation

```bash
# Check online agents
curl http://localhost:3001/gateway/demo/agents/online

# Invoke weather agent
curl -X POST http://localhost:3001/gateway/demo/agents/weather-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_weather", "parameters": {"location": "Seattle"}}'
```

### 2. PII Blocking (Governance)

Send a request containing an SSN pattern - the gateway blocks it automatically.

### 3. HITL Approval Flow

Invoke risky-agent - all operations require human approval in the console.

### 4. Rate Limiting

Rapidly invoke data-analyst to see rate limits in action.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RELAY.ONE GATEWAY (:3001)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Governance Layer                                        │   │
│  │  • PII Detection    • Rate Limiting    • Trust Levels    │   │
│  │  • HITL Triggers    • Audit Logging    • ACL Rules       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────┴────────┐  ┌────────┴────────┐  ┌───────┴────────┐
│  weather-agent  │  │  data-analyst   │  │  risky-agent   │
│  Trust: HIGH    │  │  Trust: MEDIUM  │  │  Trust: LOW    │
│  :4001          │  │  :4002          │  │  :4003         │
└─────────────────┘  └─────────────────┘  └────────────────┘
```

---

## Core Features (All Implemented)

| Feature | Status | Description |
|---------|--------|-------------|
| Agent Registry | ✅ Real | Full lifecycle management |
| Gateway Relay | ✅ Real | Routes traffic with governance |
| PII Detection | ✅ Real | Blocks SSN, credit cards, etc. |
| Rate Limiting | ✅ Real | Per-agent/org limits |
| HITL Approvals | ✅ Real | Human approval workflows |
| Trust Scoring | ✅ Real | Reputation tracking |
| Audit Logging | ✅ Real | All requests logged to MongoDB |
| Certificates | ✅ Real | mTLS for agent identity |
| Billing | ✅ Real | Stripe integration |
| Peering | ✅ Real | Cross-org agent discovery |

---

## What's Built

### Applications (10 Total)

| App | Description |
|-----|-------------|
| **api** | Core Fastify API (50+ routes, 130+ services) |
| **gateway-rust** | High-performance Rust gateway (100k+ RPS) |
| **console** | Customer portal (33 pages) |
| **admin** | Admin portal |
| **mcp** | MCP protocol server |
| **relay-one-web** | Central SaaS portal |
| **relay-one-api** | Central registry API |
| **discovery-mcp** | Agent discovery MCP server |
| **reputation-mcp** | Reputation MCP server |
| **demo-agents** | Demo agent servers |

### Packages (7 Total)

| Package | Description |
|---------|-------------|
| **@relay-one/sdk** | TypeScript SDK (14 modules) |
| **@relay-one/types** | Shared TypeScript types |
| **@relay-one/database** | MongoDB client |
| **@relay-one/ui** | React UI components |
| **@relay-one/config** | Shared configs |
| **sdk-python** | Python SDK |
| **sdk-rust** | Rust SDK |
| **sdk-go** | Go SDK |

### RelayChain (28 Rust Crates)

Purpose-built blockchain for agent payments and identity with 81,000+ lines of Rust:

- `relay-chain-core`, `crypto`, `state`, `consensus`, `network`
- `relay-chain-node`, `rpc`, `cli`, `fees`, `ownership`
- `relay-chain-payments`, `quotes`, `ratings`, `bridge`
- `relay-chain-compliance`, `quorum`, `tee`, `gateway`
- `relay-chain-backup`, `loadbalancer`, `supervisor`, `integrity`
- `relay-chain-a2a`, `capabilities`, `policy`, `reputation`

---

## SDK Usage

```bash
npm install @relay-one/sdk
```

```typescript
import { RelayClient } from '@relay-one/sdk';

const client = new RelayClient({
  apiKey: 'your-api-key',
  baseUrl: 'http://localhost:3001',
});

// Invoke agent through gateway
const result = await client.gatewayInvoke('agent-id', {
  tool: 'analyze-data',
  input: { dataset: 'sales-q4' },
});
```

---

## Key Differentiators

### 1. Real Gateway (Not a Wrapper)
- Every agent-to-agent call transits through the relay
- Policies evaluated in real-time before forwarding
- Full audit trail for compliance

### 2. Trust Levels
- `verified` - Highest trust, full access
- `high` - Verified partners
- `medium` - Standard external agents
- `low` - Probationary
- `untrusted` - Quarantined/suspicious

### 3. HITL (Human-in-the-Loop)
- High-risk operations pause for human approval
- Configurable risk thresholds
- Approval workflow in Console UI

### 4. Multi-Protocol
- REST API
- MCP (Model Context Protocol)
- A2A (Agent-to-Agent)
- WebSocket (real-time)
- RelayChain (blockchain)

---

## Gateway 7-Step Pipeline

Every invocation produces detailed trace logs:

```
[1/7] INCOMING REQUEST - Caller, target, tool, input
[2/7] AGENT LOOKUP - Find agent, check status
[3/7] GOVERNANCE CHECKS - PII scan, rate limits
[4/7] RISK ASSESSMENT - HITL evaluation
[5/7] AGENT CONNECTION - Verify online status
[6/7] INVOKING AGENT - Forward request
[7/7] POST-PROCESSING - Update reputation, audit log
```

---

## Deployment Models

### Appliance (Self-Hosted)
- Complete data sovereignty
- On-premise or private cloud
- Docker Compose or Kubernetes

### relay.one Central (SaaS)
- Hosted at relay.one
- Agent registry & discovery
- Certificate authority
- Reputation system

### Hybrid
- Self-hosted appliance + central sync
- Federated peering model

---

## Technology Stack

- **Backend**: Fastify 4.24, TypeScript 5.3, Node.js 20+
- **Gateway**: Rust, Axum, Tokio (100k+ RPS)
- **Frontend**: Next.js 14, React 18, Radix UI
- **Database**: MongoDB 6.3+, Redis 7.0
- **Auth**: JWT + mTLS certificates
- **Billing**: Stripe SDK
- **Monitoring**: Prometheus, Grafana, OpenTelemetry

---

## Scale Metrics

| Metric | Value |
|--------|-------|
| Applications | 10 |
| Packages | 7 |
| API Routes | 50+ |
| API Services | 130+ |
| Rust Crates | 24 |
| Test Files | 60+ |
| Lines of Code | 70,000+ |

---

## Environment Variables

```bash
# Required
JWT_SECRET=your-secret-key

# Optional
MONGODB_URI=mongodb://localhost:27017/relay_one
STRIPE_SECRET_KEY=sk_...
```

---

## Next Steps

1. **Run locally** - Follow 60-second start above
2. **Explore console** - http://localhost:3000
3. **Read API docs** - http://localhost:3001/docs
4. **Try the SDK** - See packages/sdk/README.md
5. **Review architecture** - See ARCHITECTURE.md

---

*Production-ready enterprise agent governance platform*
