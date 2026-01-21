# relay.one Architecture

## Overview

relay.one is a comprehensive enterprise AI agent infrastructure platform with two distinct deployment models:

1. **Appliance** - Self-hosted enterprise deployment for agentic gateway infrastructure with complete data sovereignty
2. **relay.one Central** - SaaS portal for central registry, discovery, certificates, and reputation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           relay.one Central (SaaS)                               │
│                        https://relay.one (Vercel)                                │
│                                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Web Portal │  │ Central API │  │ Discovery   │  │ Reputation/Trust        │ │
│  │  (Next.js)  │  │  (Node.js)  │  │ MCP Server  │  │ MCP Server              │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                                  │
│  Features:                                                                       │
│  - Organization profiles & license management                                    │
│  - Agent registry (public discovery)                                             │
│  - Certificate authority (identity verification)                                 │
│  - Reputation database (agent ratings, trust scores)                             │
│  - External developer portal (pay-per-use agent access)                          │
│  - Billing agreements & invoicing                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Optional sync
                                    │ (organizations opt-in)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Enterprise Appliances (On-Prem)                           │
│                                                                                  │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐     │
│  │      Organization A             │    │      Organization B             │     │
│  │   (Private Cloud / K8s)         │    │   (On-Premise / Docker)         │     │
│  │                                 │    │                                 │     │
│  │  ┌─────────┐ ┌─────────┐       │    │  ┌─────────┐ ┌─────────┐       │     │
│  │  │ Console │ │   API   │       │    │  │ Console │ │   API   │       │     │
│  │  │ :3000   │ │  :3001  │       │◄──►│  │ :3000   │ │  :3001  │       │     │
│  │  └─────────┘ └─────────┘       │    │  └─────────┘ └─────────┘       │     │
│  │  ┌─────────────────────────┐   │    │  ┌─────────────────────────┐   │     │
│  │  │ Gateway Rust :3100      │   │    │  │ Gateway Rust :3100      │   │     │
│  │  │ (100k+ RPS)             │   │    │  │ (100k+ RPS)             │   │     │
│  │  └─────────────────────────┘   │    │  └─────────────────────────┘   │     │
│  │  ┌─────────┐ ┌─────────┐       │    │  ┌─────────┐ ┌─────────┐       │     │
│  │  │ MongoDB │ │  Redis  │       │    │  │ MongoDB │ │  Redis  │       │     │
│  │  └─────────┘ └─────────┘       │    │  └─────────┘ └─────────┘       │     │
│  │                                 │    │                                 │     │
│  │  Private agents, complete       │    │  Independent deployment,       │     │
│  │  data sovereignty               │    │  optional peering              │     │
│  └─────────────────────────────────┘    └─────────────────────────────────┘     │
│                  │                              │                                │
│                  └──────────── Peering ─────────┘                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Platform Components

### Applications (12 Total)

| App | Technology | Port | Description |
|-----|------------|------|-------------|
| **api** | Fastify 4.24 | 3001 | Core API with 71 routes, 124 services |
| **gateway-rust** | Rust/Axum/Tokio | 3100 | High-performance gateway (100k+ RPS) |
| **console** | Next.js 14 | 3000 | Customer dashboard (61 pages) |
| **admin** | Next.js 14 | 3002 | Internal admin portal (25 pages) |
| **mcp** | Node.js | - | MCP server for appliance AI integration |
| **relay-one-web** | Next.js 14 | Vercel | Central SaaS marketing + portal |
| **relay-one-api** | Fastify 4.26 | Vercel | Central registry API |
| **discovery-mcp** | Node.js | - | Agent discovery MCP server |
| **reputation-mcp** | Node.js | - | Reputation data MCP server |
| **demo-agents** | Fastify 4.24 | 5001-5003 | Weather, analyst, risky demo agents |

### Packages (10 Total)

| Package | Description |
|---------|-------------|
| **@relay-one/sdk** | TypeScript SDK with 13 modules (client, agent, a2a, analytics, behavior-auth, causality, crypto, federation, governance, payments, quotes, ratings, relaychain) |
| **@relay-one/types** | Shared TypeScript type definitions |
| **@relay-one/database** | MongoDB client with connection pooling, custom errors, structured logging |
| **@relay-one/ui** | React component library (20+ Radix UI + Tailwind components) |
| **@relay-one/config** | Shared ESLint, Prettier, Tailwind configs |
| **sdk-python** | Python SDK with capabilities and relaychain support |
| **sdk-rust** | Rust SDK bindings with comprehensive test coverage |
| **sdk-go** | Go SDK with full agent, payment, governance, and relaychain support |

### RelayChain Blockchain (28 Rust Crates)

Purpose-built blockchain for agent payments and identity:

| Crate | Lines | Description |
|-------|-------|-------------|
| `relay-chain-core` | 72 | Core types and traits |
| `relay-chain-crypto` | 236 | Ed25519, BLAKE3 cryptography |
| `relay-chain-state` | 54 | State management |
| `relay-chain-consensus` | 47 | Consensus mechanism |
| `relay-chain-network` | 49 | P2P networking |
| `relay-chain-node` | 300 | Full node implementation |
| `relay-chain-rpc` | 72 | JSON-RPC server |
| `relay-chain-cli` | 286 | Command-line interface |
| `relay-chain-fees` | 88 | Fee calculation |
| `relay-chain-ownership` | 89 | Asset ownership |
| `relay-chain-payments` | 123 | Payment processing |
| `relay-chain-quotes` | 89 | Quote system |
| `relay-chain-ratings` | 86 | Reputation ratings |
| `relay-chain-bridge` | 990 | Cross-chain bridge (largest) |
| `relay-chain-compliance` | 637 | Regulatory compliance |
| `relay-chain-testnet` | 82 | Test network utilities |
| `relay-chain-orchestrator` | 114 | Orchestration |
| `relay-chain-gateway` | 103 | Gateway interface |
| `relay-chain-eth-rpc` | 128 | Ethereum RPC compatibility |
| `relay-chain-integrity` | 106 | Data integrity checks |
| `relay-chain-tee` | 246 | Trusted Execution Environment |
| `relay-chain-supervisor` | 148 | Node supervision |
| `relay-chain-quorum` | 619 | Quorum voting (second largest) |
| `relay-chain-loadbalancer` | 154 | Load balancing |
| `relay-chain-backup` | 281 | Backup, recovery, and AES-256-GCM encryption |
| `relay-chain-auth` | 150+ | Authentication and authorization |
| `relay-chain-metrics` | 120+ | Metrics collection and reporting |
| `relay-chain-tracing` | 100+ | Distributed tracing support |

**Total RelayChain:** ~6,000+ lines of Rust across 28 crates

---

## Directory Structure

```
relay.one/
├── apps/
│   │
│   │  ════════════════════════════════════════════════════════════
│   │  APPLIANCE (Self-hosted on-prem/private cloud deployment)
│   │  ════════════════════════════════════════════════════════════
│   │
│   ├── api/                    # Core API server (Fastify)
│   │   ├── src/
│   │   │   ├── routes/         # 50+ route handlers
│   │   │   ├── services/       # 130+ business logic services (~3MB)
│   │   │   └── middleware/     # Auth, validation, errors
│   │   └── Dockerfile
│   │
│   ├── gateway-rust/           # High-performance gateway (Rust)
│   │   ├── src/
│   │   │   ├── main.rs         # Entry point (~300 lines)
│   │   │   ├── config/         # Configuration management
│   │   │   ├── handlers/       # HTTP route handlers
│   │   │   ├── services/       # Rate limiting, PII detection
│   │   │   ├── middleware/     # Auth, tracing
│   │   │   └── models/         # Data models
│   │   ├── Cargo.toml
│   │   └── Dockerfile
│   │
│   ├── console/                # Customer dashboard (Next.js)
│   │   ├── src/app/
│   │   │   ├── agents/         # Agent management
│   │   │   ├── governance/     # Policy management
│   │   │   ├── certificates/   # Certificate management
│   │   │   ├── relaychain/     # Blockchain UI (faucet, network, agents)
│   │   │   ├── billing/        # Usage and billing
│   │   │   ├── reports/        # Analytics and reports
│   │   │   └── ...             # 33 total pages
│   │   └── Dockerfile
│   │
│   ├── admin/                  # Internal admin portal (Next.js)
│   │   └── src/app/            # Platform administration
│   │
│   ├── mcp/                    # MCP protocol server
│   │   └── src/index.ts        # 102KB MCP implementation
│   │
│   ├── demo-agents/            # Demo agent servers
│   │   ├── weather-agent.ts    # Weather data (port 4001)
│   │   ├── data-analyst.ts     # Data analysis (port 4002)
│   │   └── risky-agent.ts      # HITL demo (port 4003)
│   │
│   │  ════════════════════════════════════════════════════════════
│   │  RELAY.ONE CENTRAL (SaaS hosted at relay.one)
│   │  ════════════════════════════════════════════════════════════
│   │
│   ├── relay-one-web/          # Central SaaS web portal (Next.js/Vercel)
│   │   └── src/app/
│   │       ├── (marketing)/    # Public marketing pages
│   │       ├── (auth)/         # Login/signup
│   │       ├── dashboard/      # Organization dashboard
│   │       ├── registry/       # Agent registry browser
│   │       └── developers/     # External developer portal
│   │
│   ├── relay-one-api/          # Central SaaS API (Vercel Functions)
│   │   └── src/
│   │       ├── routes/
│   │       │   ├── registry/   # Central agent registry
│   │       │   ├── discovery/  # Agent discovery API
│   │       │   ├── certs/      # Certificate authority
│   │       │   ├── reputation/ # Trust and reputation
│   │       │   └── billing/    # SaaS billing
│   │       └── services/
│   │
│   ├── discovery-mcp/          # Discovery MCP server
│   │   └── src/index.ts        # 14.5KB - Registry search via MCP
│   │
│   └── reputation-mcp/         # Reputation MCP server
│       └── src/index.ts        # 19.7KB - Trust scores via MCP
│
├── packages/
│   ├── sdk/                    # TypeScript SDK
│   │   └── src/
│   │       ├── client.ts       # Main RelayClient
│   │       ├── agent.ts        # RelayAgent base class
│   │       ├── a2a.ts          # Agent-to-Agent messaging
│   │       ├── analytics.ts    # Analytics tracking
│   │       ├── behavior-auth.ts # Behavioral authentication
│   │       ├── causality.ts    # Event causality tracking
│   │       ├── crypto.ts       # Certificate signing
│   │       ├── federation.ts   # Federation peer management
│   │       ├── governance.ts   # Policy governance
│   │       ├── payments.ts     # Payment protocols
│   │       ├── quotes.ts       # Quote negotiation
│   │       ├── ratings.ts      # Reputation ratings
│   │       └── relaychain/     # Blockchain integration (11 files)
│   │
│   ├── sdk-python/             # Python SDK
│   │   └── relay_sdk/
│   │       ├── capabilities.py
│   │       └── relaychain.py
│   │
│   ├── sdk-rust/               # Rust SDK
│   │   └── src/
│   │       ├── lib.rs
│   │       └── relaychain/mod.rs
│   │
│   ├── database/               # MongoDB client
│   │   └── src/
│   │       ├── client.ts       # Connection pooling
│   │       ├── errors.ts       # Custom error classes
│   │       └── logger.ts       # Structured logging
│   │
│   ├── types/                  # Shared TypeScript types
│   │   └── src/index.ts        # 2,226+ lines of types
│   │
│   ├── ui/                     # React UI components
│   │   └── src/                # 20+ Radix UI components
│   │
│   └── config/                 # Shared configurations
│
├── relay-chain/                # Blockchain implementation
│   ├── Cargo.toml              # Workspace definition
│   └── crates/                 # 28 Rust crates
│
├── deploy/
│   ├── docker/
│   │   ├── docker-compose.yml          # Production stack
│   │   ├── docker-compose.dev.yml      # Development with hot reload
│   │   └── .env.example
│   │
│   ├── kubernetes/
│   │   └── helm/
│   │       └── relay-one-appliance/    # Helm chart
│   │           ├── Chart.yaml
│   │           ├── values.yaml
│   │           └── templates/
│   │
│   └── digitalocean/                   # DO App Platform specs
│
├── tests/                      # Test suites (60+ files)
│   ├── api/                    # API service tests
│   ├── database/               # Database tests
│   ├── sdk/                    # SDK tests
│   ├── security/               # Security tests
│   └── integration/            # Integration tests
│
└── docs/                       # Documentation
    ├── IAM.md                  # IAM system
    ├── CAPABILITIES.md         # Capabilities system
    └── flowcore/               # Flowcore integration
```

---

## Appliance vs Central Feature Matrix

| Feature | Appliance (On-Prem) | relay.one Central |
|---------|---------------------|-------------------|
| Agent hosting | ✅ Local agents | ❌ Registry only |
| Agent invocation | ✅ Gateway routes | ❌ Discovery only |
| MongoDB data | ✅ Local/private | ✅ Central DB |
| Data sovereignty | ✅ Complete | ❌ Shared |
| Agent registry | ✅ Local registry | ✅ Global registry |
| Public discovery | ❌ Private | ✅ Public API |
| Certificate issuance | ✅ Self-signed or relay.one | ✅ Central CA |
| Reputation tracking | ✅ Local scores | ✅ Global scores |
| Peering | ✅ Direct P2P | ✅ Via discovery |
| External developers | ✅ Local billing | ✅ Central billing |
| Billing/invoicing | ✅ Self-managed | ✅ Stripe Connect |
| License management | ❌ N/A | ✅ License server |

---

## Data Flow Patterns

### Appliance Only (Private Deployment)

```
[User] → [Console] → [API] → [MongoDB]
                        ↓
                   [Gateway Rust] → [Agents]
```

### Appliance + Central Registry (Hybrid)

```
[User] → [Console] → [API] → [MongoDB]
                        ↓
                   [Gateway Rust] → [Agents]
                        ↓
              [Sync to Central Registry]
                        ↓
              [relay.one Discovery API]
                        ↓
              [External Developers]
```

### External Developer Flow

```
[External Dev] → [relay.one Portal] → [Create Account]
                                            ↓
                                    [Add Payment Method]
                                            ↓
                                    [Get Payment Key]
                                            ↓
[External Dev] → [relay.one API] → [Resolve Agent Location]
                                            ↓
              [Direct to Appliance Gateway with Payment Key]
                                            ↓
                             [Appliance validates & bills]
```

### Gateway 7-Step Pipeline

Every agent invocation follows this pipeline:

```
[1/7] INCOMING REQUEST - Caller, target, tool, input
[2/7] AGENT LOOKUP - Find agent, check status
[3/7] GOVERNANCE CHECKS - PII scan, rate limits, policies
[4/7] RISK ASSESSMENT - HITL evaluation
[5/7] AGENT CONNECTION - Verify online status
[6/7] INVOKING AGENT - Forward request to agent
[7/7] POST-PROCESSING - Update reputation, audit log
```

---

## Unified Capabilities System

Protocol-agnostic framework for registering, discovering, and invoking agent capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Capabilities Architecture                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Protocols                 Registry                    Integration             │
│  ┌─────────────┐      ┌─────────────────────┐      ┌─────────────────────────┐ │
│  │     MCP     │──────│                     │──────│    Policy Engine        │ │
│  └─────────────┘      │                     │      └─────────────────────────┘ │
│  ┌─────────────┐      │    Capabilities     │      ┌─────────────────────────┐ │
│  │     A2A     │──────│      Registry       │──────│  Reputation System      │ │
│  └─────────────┘      │                     │      └─────────────────────────┘ │
│  ┌─────────────┐      │  • Semantic Search  │      ┌─────────────────────────┐ │
│  │    REST     │──────│  • Vector Embeddings│──────│   Billing Ledger        │ │
│  └─────────────┘      │  • Access Control   │      └─────────────────────────┘ │
│  ┌─────────────┐      │  • Pricing Models   │      ┌─────────────────────────┐ │
│  │ RelayChain  │──────│  • Metrics & Stats  │──────│   Audit Logging         │ │
│  └─────────────┘      └─────────────────────┘      └─────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Supported Protocols

| Protocol | Description | Authentication | Use Case |
|----------|-------------|----------------|----------|
| MCP | Model Context Protocol | Bearer, API Key | AI model integration |
| A2A | Agent-to-Agent | Certificate, Signature | Autonomous agents |
| REST | RESTful HTTP | API Key, OAuth2 | Standard APIs |
| WebSocket | Real-time bidirectional | Bearer | Streaming |
| gRPC | High-performance RPC | Certificate | Microservices |
| RelayChain | Blockchain on-chain | Wallet signature | Payments |

### Well-Known Discovery Endpoints

| Endpoint | Protocol | Response |
|----------|----------|----------|
| `/.well-known/mcp-servers` | MCP | Available MCP servers |
| `/.well-known/a2a-agents` | A2A | Available A2A agents |
| `/.well-known/relay-agents` | RelayChain | On-chain agents |

### Capability Categories (19)

- **Text:** generation, analysis, translation
- **Image:** generation, analysis
- **Audio:** generation, analysis
- **Video:** generation, analysis
- **Code:** generation, analysis
- **Data:** analysis, search, retrieval
- **AI:** reasoning, memory, embedding, tool-use, moderation
- **Custom:** domain-specific capabilities

---

## Security Model

### Appliance Security

- All data stored locally (complete sovereignty)
- mTLS between services (optional)
- JWT authentication for users
- API keys for programmatic access
- Certificate-based agent identity
- Quantum-safe cryptography (NIST PQC)

### Central Security

- OAuth 2.0 for organization login
- Payment keys for external developers
- Signing keys for billing agreements
- Certificate fingerprint verification
- Rate limiting per key

### Trust Hierarchy

```
relay.one CA (Root of Trust)
    │
    ├── Organization Certificate
    │       │
    │       └── Agent Certificate
    │               │
    │               └── Request Signature
    │
    └── External Developer
            │
            └── Payment Key
```

---

## API Boundaries

### Appliance API (`/api/v1/*`, `/gateway/*`)

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Agents | `/api/v1/agents/*` | Agent CRUD, lifecycle |
| Gateway | `/gateway/*` | Agent invocation, connections |
| Governance | `/governance/*` | Policy management |
| HITL | `/hitl/*` | Approval workflows |
| Certificates | `/api/v1/certificates/*` | mTLS certificates |
| Peering | `/api/v1/peering/*` | Organization peering |
| Billing | `/billing/*` | Usage and billing |
| Capabilities | `/api/v1/capabilities/*` | Semantic discovery |
| A2A | `/a2a/*` | Agent-to-Agent messaging |
| Compliance | `/compliance/*` | Regulatory controls |

### Central API (`/v1/*`)

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Registry | `/v1/registry/*` | Agent registration |
| Discovery | `/v1/discovery/*` | Public agent search |
| Certificates | `/v1/certificates/*` | CA certificate issuance |
| Reputation | `/v1/reputation/*` | Trust scores and ratings |
| Developers | `/v1/developers/*` | External developer management |
| Billing | `/v1/billing/*` | SaaS billing |
| Organizations | `/v1/organizations/*` | Org profile management |

---

## MCP Servers

### Discovery MCP Server (`apps/discovery-mcp`)

Exposes agent discovery to AI assistants via Model Context Protocol.

**Tools:**
| Tool | Description | Auth Required |
|------|-------------|---------------|
| `search_agents` | Search agents by name, capabilities, or description | No |
| `get_agent` | Get detailed info about a specific agent | No |
| `list_capabilities` | List all available capabilities with counts | No |
| `get_agent_reputation` | Get reputation score for an agent | No |
| `invoke_agent` | Call a tool on a remote agent | Yes |

**Resources:**
- `relay://registry/stats` - Current registry statistics

### Reputation MCP Server (`apps/reputation-mcp`)

Exposes trust and reputation data to AI assistants.

**Tools:**
| Tool | Description | Auth Required |
|------|-------------|---------------|
| `get_agent_reputation` | Get trust score, success rate, and event counts | No |
| `get_reputation_history` | Get reputation events over time | No |
| `get_leaderboard` | Get top agents ranked by reputation | No |
| `compare_agents` | Compare multiple agents side-by-side | No |
| `report_agent` | Report an agent for misconduct | Yes |
| `get_reputation_stats` | Get global reputation statistics | No |
| `is_agent_trusted` | Quick trust check with configurable thresholds | No |

**Resources:**
- `relay://reputation/stats` - Global reputation statistics
- `relay://reputation/leaderboard` - Top agents by reputation

---

## Deployment Options

### Appliance Deployment

#### Docker Compose (Single Node)

```bash
cd deploy/docker
docker-compose up -d
```

#### Kubernetes (Helm)

```bash
helm install relay-one ./deploy/kubernetes/helm/relay-one-appliance \
  --namespace relay-one \
  --create-namespace \
  --values custom-values.yaml
```

#### DigitalOcean App Platform

```bash
doctl apps create --spec deploy/digitalocean/app-spec.yaml
```

### relay.one Central Deployment

Deployed on Vercel:

```bash
# Web portal
cd apps/relay-one-web && vercel --prod

# API
cd apps/relay-one-api && vercel --prod
```

---

## Performance Benchmarks

### Rust Gateway vs Node.js

| Metric | Node.js Gateway | Rust Gateway | Improvement |
|--------|-----------------|--------------|-------------|
| Requests/sec | ~10,000 | ~100,000+ | **10x** |
| P50 Latency | ~15ms | ~1-2ms | **7-10x** |
| P99 Latency | ~100ms | ~10ms | **10x** |
| Memory Usage | ~500MB | ~50MB | **10x** |
| CPU at 10k RPS | 100% | 20% | **5x** |

---

## Version Compatibility

| Component | Min Version | Recommended |
|-----------|-------------|-------------|
| Node.js | 20.x | 20.x |
| Rust | 1.75+ | 1.75+ |
| MongoDB | 6.0 | 7.0 |
| Redis | 7.0 | 7.0 |
| Kubernetes | 1.28 | 1.29 |
| Docker | 24.0 | 24.0 |

---

## Scale Metrics

| Metric | Value |
|--------|-------|
| **Total Apps** | 12 (9 Node.js + 1 Rust + 2 MCP servers) |
| **Total Packages** | 10 (7 TypeScript + 3 language SDKs) |
| **API Routes** | 71 endpoint files |
| **API Services** | 124 service files |
| **Database Adapters** | 4 SQL + 1 NoSQL + Data Lake support |
| **LLM Providers** | 4 (OpenAI, Azure, Bedrock, Vertex) |
| **Rust Crates** | 28 (6,000+ LOC) |
| **Test Files** | 60+ (comprehensive coverage) |
| **Documentation** | 17+ MD files |
| **UI Components** | 20+ shared Radix UI components |
| **SDK Modules** | 13 core TypeScript + Python + Rust + Go |
