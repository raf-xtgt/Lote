# Lote: Setup

Last updated: 2026-10-29. Verified on Linux Mint 22.3 (Ubuntu 24.04 base), bash. Values in `<ANGLE_BRACKETS>` are
yours to fill in.

## One-Time Setup

### 1. Tools and versions
| Tool | Version | Install / check |
|---|---|---|
| uv | 0.12.19 | `curl -LsSf https://astral.sh/uv/install.sh \| sh`; `uv --version` |
| Python | 3.13.x (uv downloads it on `uv sync`) | `uv run python --version` |
| Node | 24 (`frontend/.nvmrc`) | `nvm install 24`; `node --version` |
| npm | 11.x | `npm --version` |
| Modal CLI | 1.6.3 (installed by `uv sync` in `backend/.venv`) | `uv run modal --version` |
| Playwright Chromium (local) | from Playwright 1.64.1 | `uv run playwright install --with-deps chromium` (asks for sudo) |
| Tiled (optional, to open exports) | 1.12.x | https://www.mapeditor.org |

### 2. Accounts and services
| Service | What for | Where to get the value | Value |
|---|---|---|---|
| Meta Model API | Muse Image, SAM 3.1, Muse Spark, Muse Wonder | https://dev.meta.ai → API keys | `<YOUR_META_MODEL_API_KEY>` |
| Modal | Hosting (web, runs, Blender, Chromium), Volume, Dicts, Secret | https://modal.com | workspace `<YOUR_MODAL_WORKSPACE>` |
| itch.io (optional) | Sample player builds | https://itch.io/dashboard | `<YOUR_ITCH_PAGE>` |

### 3. Get the code and install dependencies
```bash
git clone <REPO_URL>
cd Lote/backend && uv sync
cd ../frontend && npm install
```

### 4. Configuration and secrets
```bash
cd Lote/backend
cp .env.example .env     # set MODEL_API_KEY=<YOUR_META_MODEL_API_KEY>
```
Modal login and cloud resources (once per Modal account):
```bash
cd Lote/backend
uv run modal setup
uv run modal secret create meta-model-api --from-dotenv .env
uv run modal volume create lote-runs --version=2
uv run modal dict create lote-progress
uv run modal dict create lote-spend
```

### 5. Assets
CC0 Quaternius models used for pose guides are in `backend/assets/quaternius/` (sources in
`backend/assets/LICENSE.md`).

## Run Locally
Terminal 1:
```bash
cd Lote/backend
uv run modal serve modal_app.py
```
Prints a dev URL like `https://<YOUR_MODAL_WORKSPACE>--lote-web-dev.<region>.modal.run`.

Terminal 2:
```bash
cd Lote/frontend
LOTE_API_URL=https://<YOUR_MODAL_WORKSPACE>--lote-web-dev.<region>.modal.run npm run dev
```
Opens `http://localhost:5173/tool/` ("API health: ok") and `/player/`.

### Run the tests and checks
```bash
cd Lote/backend
uv run pytest -q
uv run ruff check .
uv run modal run modal_app.py::smoke                          # 3 x [OK]: storage, guide, playtest
uv run --env-file .env python scripts/day1_meta_check.py      # Meta API check (~$0.10)

cd ../frontend
npm run typecheck
npm run build
```

## Deploy
```bash
cd Lote/frontend && npm run build
cd ../backend && uv run modal deploy modal_app.py
```
Live URL: `https://<YOUR_MODAL_WORKSPACE>--lote-web.<region>.modal.run`. Check `/api/health` returns
`{"status":"ok",...}` and `/player/` loads. Roll back by redeploying an earlier commit. Stop with
`uv run modal app stop lote`.
