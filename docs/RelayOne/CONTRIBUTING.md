# Contributing to relay.one

Thank you for your interest in contributing to relay.one! This guide will help you get started.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)

---

## Development Setup

### Prerequisites

| Tool | Version | Installation |
|------|---------|--------------|
| Node.js | 20.x | [nodejs.org](https://nodejs.org) |
| pnpm | 8.x | `npm install -g pnpm` |
| Docker | 24.x | [docker.com](https://docker.com) |
| Rust | 1.75+ | [rustup.rs](https://rustup.rs) (for gateway-rust) |

### Quick Setup

```bash
# Clone repository
git clone https://github.com/relay-one/relay-one.git
cd relay-one

# Install dependencies
pnpm install

# Start infrastructure
docker run -d -p 27017:27017 --name relay-mongo mongo:7
docker run -d -p 6379:6379 --name relay-redis redis:7

# Seed demo data
cd scripts && npx tsx seed-db.ts && cd ..

# Start development servers
pnpm dev
```

### Environment Configuration

Create `.env.local` in the project root:

```bash
JWT_SECRET=dev-secret-change-in-production
MONGODB_URI=mongodb://localhost:27017/relay_one
REDIS_URL=redis://localhost:6379
```

---

## Project Structure

```
relay.one/
├── apps/                       # Applications
│   ├── api/                    # Fastify API server
│   │   ├── src/
│   │   │   ├── routes/         # 71 route files
│   │   │   ├── services/       # 121 service files
│   │   │   └── middleware/     # Auth, validation
│   │   └── package.json
│   │
│   ├── gateway-rust/           # Rust gateway (100k+ RPS)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── handlers/
│   │   │   ├── services/
│   │   │   └── middleware/
│   │   └── Cargo.toml
│   │
│   ├── console/                # Customer dashboard (Next.js)
│   │   └── src/app/            # 33+ pages
│   │
│   ├── admin/                  # Admin portal (Next.js)
│   │   └── src/app/            # 23+ pages
│   │
│   ├── relay-one-web/          # Marketing site (Next.js)
│   ├── relay-one-api/          # Central registry API
│   ├── discovery-mcp/          # Discovery MCP server
│   ├── reputation-mcp/         # Reputation MCP server
│   ├── mcp/                    # Appliance MCP server
│   └── demo-agents/            # Demo agent servers
│
├── packages/                   # Shared packages
│   ├── sdk/                    # TypeScript SDK
│   ├── sdk-python/             # Python SDK
│   ├── sdk-rust/               # Rust SDK
│   ├── sdk-go/                 # Go SDK
│   ├── types/                  # Shared TypeScript types
│   ├── database/               # MongoDB client
│   ├── ui/                     # React UI components
│   └── config/                 # Shared configs
│
├── relay-chain/                # Blockchain (28 Rust crates)
│   └── crates/
│
├── tests/                      # Test suites
├── docs/                       # Documentation
└── deploy/                     # Deployment configs
```

### Key Files

| File | Purpose |
|------|---------|
| `pnpm-workspace.yaml` | Workspace configuration |
| `turbo.json` | Turborepo build configuration |
| `package.json` | Root scripts and dependencies |
| `.env.example` | Environment template |

---

## Development Workflow

### Running Services

```bash
# All services
pnpm dev

# Individual services
pnpm dev:api        # API on :3001
pnpm dev:console    # Console on :3000
pnpm dev:admin      # Admin on :3002

# Demo agents
cd apps/demo-agents && pnpm dev
```

### Building

```bash
# Build all
pnpm build

# Build specific app
pnpm --filter @relay-one/api build

# Build Rust gateway
cd apps/gateway-rust && cargo build --release
```

### Type Checking

```bash
pnpm typecheck
```

---

## Code Standards

### TypeScript

- Use strict TypeScript (`"strict": true`)
- Add JSDoc comments for public APIs
- Use Zod for runtime validation
- Prefer `const` over `let`
- Use meaningful variable names

```typescript
/**
 * Creates a new agent in the registry.
 * @param data - Agent creation data
 * @returns The created agent
 * @throws {ValidationError} If data is invalid
 */
export async function createAgent(data: CreateAgentInput): Promise<Agent> {
  const validated = createAgentSchema.parse(data);
  // Implementation
}
```

### React/Next.js

- Use functional components with hooks
- Prefer server components where possible
- Use Tailwind CSS for styling
- Keep components small and focused

```tsx
/**
 * Agent status badge component.
 */
export function AgentStatusBadge({ status }: { status: AgentStatus }) {
  const colors = {
    active: 'bg-emerald-100 text-emerald-800',
    inactive: 'bg-slate-100 text-slate-800',
    suspended: 'bg-red-100 text-red-800',
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[status]}`}>
      {status}
    </span>
  );
}
```

### Rust

- Follow Rust idioms and conventions
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Add documentation comments

```rust
/// Handles incoming gateway requests.
///
/// # Arguments
/// * `request` - The incoming HTTP request
///
/// # Returns
/// The processed response or an error
pub async fn handle_request(request: Request) -> Result<Response, GatewayError> {
    // Implementation
}
```

### Commit Messages

Use conventional commits:

```
feat(api): add agent lifecycle management endpoints
fix(console): resolve HITL approval button state
docs(readme): update deployment instructions
test(sdk): add payment protocol tests
chore(deps): update fastify to 4.25
```

**Prefixes:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `chore`: Maintenance
- `refactor`: Code refactoring
- `style`: Formatting

---

## Testing

### Running Tests

```bash
# All tests
pnpm test

# With coverage
pnpm test:coverage

# Specific package
pnpm --filter @relay-one/api test

# Watch mode
pnpm test -- --watch
```

### Test Structure

```
tests/
├── api/                    # API service tests
│   ├── auth.test.ts
│   ├── agents.test.ts
│   └── governance.test.ts
├── sdk/                    # SDK tests
├── integration/            # Integration tests
└── e2e/                    # End-to-end tests
```

### Writing Tests

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { createAgent } from '../src/services/agent.service';

describe('Agent Service', () => {
  beforeEach(async () => {
    // Setup
  });

  it('should create an agent with valid data', async () => {
    const agent = await createAgent({
      name: 'test-agent',
      description: 'Test agent',
      capabilities: ['test'],
    });

    expect(agent).toBeDefined();
    expect(agent.name).toBe('test-agent');
    expect(agent.status).toBe('active');
  });

  it('should reject invalid agent data', async () => {
    await expect(
      createAgent({ name: '' })
    ).rejects.toThrow('Name is required');
  });
});
```

### E2E Tests

```bash
pnpm test:e2e
```

Uses Playwright for browser testing.

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following standards above
- Add tests for new functionality
- Update documentation as needed

### 3. Test Locally

```bash
pnpm test
pnpm lint
pnpm typecheck
pnpm build
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat(api): add new feature"
git push origin feat/my-feature
```

### 5. Create Pull Request

Include in PR description:

```markdown
## Summary
Brief description of changes

## Changes
- Added X
- Fixed Y
- Updated Z

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if UI changes)
[Add screenshots]
```

### 6. Code Review

- Address reviewer feedback
- Keep PR focused and reasonably sized
- Squash commits if requested

### 7. Merge

Once approved, the PR will be merged by a maintainer.

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/relay-one/relay-one/issues)
- **Discussions**: [GitHub Discussions](https://github.com/relay-one/relay-one/discussions)
- **Email**: dev@relay.one

---

## License

By contributing, you agree that your contributions will be licensed under the project's license.
