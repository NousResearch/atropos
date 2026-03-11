# Environments Hub (Web)

Next.js app for the Atropos Environments Hub. The UI now supports two modes:

- Local development mode with the existing Next route handlers under `app/api`.
- Static mirror mode for GitHub Pages, powered by generated JSON metadata plus GitHub raw file fetches.

## Setup

```bash
npm install
```

Generate the frontend data from repo root:

```bash
python scripts/build_env_manifest.py
```

That command writes:

- `web/public/environments.json`
- `web/public/env-data/*.json`

## Run

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Static Export

The GitHub Pages workflow:

1. Runs `python scripts/build_env_manifest.py`
2. Temporarily moves `web/app/api` out of the app tree
3. Builds the site with `STATIC_EXPORT=true`
4. Deploys the resulting `web/out` directory to Pages

Static pages fetch:

- Generated metadata from `public/`
- File contents from `raw.githubusercontent.com`

This keeps the hosted UI lightweight while preserving environment inspection.
