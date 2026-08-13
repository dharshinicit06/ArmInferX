# ArmInferX — Public Deployment Guide

Deploys ArmInferX so anyone can use it from a browser. Two pieces:

```
Browser ──► Frontend (Vercel, static React) ──► Backend (Hugging Face Spaces, FastAPI)
                       :3000→CDN                        llama.cpp + Q4_K_M (CPU, port 7860)
```

| Piece | Host | Why |
|---|---|---|
| Backend | **Hugging Face Spaces (Docker)** | Free 16 GB RAM / 2 vCPU CPU Basic tier, public URL, built for ML demos; can hold the 2 GB model + 2.5 GB inference peak |
| Frontend | **Vercel** | Free static hosting for the React build; CDN-served |

> **Why not Vercel for the backend?** The API runs llama.cpp with a 2 GB GGUF
> model at ~2.5 GB peak RAM and sustained CPU per generation. That is a
> long-running Python workload — Vercel is for frontend/edge, not this. HF
> Spaces gives you a real container with the RAM for free.

---

## 1. Backend — Hugging Face Spaces

Everything you need is in [`hf-space/`](hf-space/). The Space is its own git
repo (`https://huggingface.co/spaces/<user>/<space-name>`); the build context
is the Space repo root.

### 1.1 Create the Space

1. Go to https://huggingface.co/new-space
2. Name: `arminferx` (or anything), **License: MIT**, SDK: **Docker**.
3. Create → clone the Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<user>/arminferx
   cd arminferx
   ```

### 1.2 Populate the Space repo

Copy the deploy files in and sync the backend code (run in Git Bash on
Windows):

```bash
# from the ArmInferX project repo:
cp deploy/hf-space/Dockerfile deploy/hf-space/README.md deploy/hf-space/download_model.py .
bash deploy/hf-space/sync-backend.sh   # copies backend/ into the Space repo
```

Then in the Space repo:

```bash
git add -A
git commit -m "Deploy ArmInferX backend"
git push
```

HF builds the image (downloads + verifies the ~2 GB Q4_K_M GGUF), then your
Space is live at:

```
https://<user>-arminferx.hf.space
```

### 1.3 Set CORS for the frontend

In the Space → **Settings → Variables and secrets**, add:

| Key | Value |
|---|---|
| `ARMINFERX_CORS_ORIGINS` | `https://<your-project>.vercel.app` |

(Add more origins comma-separated if needed. After changing secrets, HF
restarts the Space.)

### 1.4 Verify the backend

```bash
curl -s https://<user>-arminferx.hf.space/health
# {"status":"healthy","available_engines":["llamacpp-optimized"],"loaded_engines":[],...}

curl -s -X POST https://<user>-arminferx.hf.space/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain what an AI inference engine is.","engine_id":"llamacpp-optimized"}'
# First call loads the model (~16–60 s); later calls stream back in seconds.
```

---

## 2. Frontend — Vercel

1. Import the GitHub repo at https://vercel.com/new.
2. **Root directory: `frontend`** (the `frontend/vercel.json` sets Vite build →
   `dist`).
3. Add the build-time env var (Settings → Environment Variables, then redeploy):

   | Key | Value |
   |---|---|
   | `VITE_API_URL` | `https://<user>-arminferx.hf.space` |

   The built JS reads `VITE_API_URL` (`frontend/src/App.jsx`), defaulting to
   `http://localhost:8000` — so this value must be set **before** the build.
4. Deploy → your dashboard is live at `https://<project>.vercel.app`.

> The dashboard falls back to the bundled optimization report
> (`frontend/public/optimization-report.json`) when the API is unreachable,
> so the page renders even during a Space cold start.

---

## 3. Wire it together

| Setting | Where | Value |
|---|---|---|
| `VITE_API_URL` | Vercel env var | `https://<user>-arminferx.hf.space` |
| `ARMINFERX_CORS_ORIGINS` | HF Space secret | `https://<project>.vercel.app` |

Open `https://<project>.vercel.app` → Studio (streaming + benchmark panel)
and Optimization Dashboard both work against the live API.

---

## Cost, limits & notes

- **Free tier:** HF Spaces CPU Basic = 2 vCPU / 16 GB RAM / 50 GB disk,
  public URL. The Space **sleeps after ~48 h of inactivity** and wakes on the
  next visit (adds ~30–60 s cold start). Vercel free tier is fine for a demo.
- **Performance:** 2 vCPU is slower than the 8-thread dev laptop; expect
  longer generations than the 5.77 tok/s measured locally. The dashboard
  reports whatever the API measures.
- **No auth:** the API has no authentication. Fine for a hackathon demo; do
  not put sensitive data behind it.
- **Persistence:** benchmark records write to `/app/results` inside the
  container (ephemeral on Spaces). Everything the dashboard needs also ships
  in the bundled static report, so this is non-blocking.
- **Rebuilds:** every push to the Space repo rebuilds the image; the 2 GB
  download is cached in the build layer when unchanged.
