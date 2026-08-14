---
title: ArmInferX
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - inference
  - llama-cpp
  - fastapi
  - qwen
  - cpu
---

# ArmInferX — AI Inference Optimization Studio

This Space hosts the **ArmInferX backend** (FastAPI + llama.cpp, CPU-only)
behind a public URL. The Q4_K_M GGUF is downloaded and SHA-256-verified
during the image build, so the API is ready to serve on the first request.

The interactive dashboard (chat Studio + Optimization Dashboard) is the
React frontend, deployed separately (e.g. Vercel) with `VITE_API_URL`
pointing at this Space:

```
https://<your-username>-arminferx.hf.space
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + engine state |
| `POST` | `/generate` | `{"prompt": "..."}` → text + latency, TTFT, tokens/sec |
| `POST` | `/generate/stream` | SSE token stream |
| `GET` | `/benchmarks` | Saved benchmark records |
| `GET` | `/optimization/report` | Measured evidence report |
| `GET` | `/docs` | Swagger UI |

Try it:

```bash
curl -s https://<your-username>-arminferx.hf.space/health
curl -s -X POST https://<your-username>-arminferx.hf.space/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain what an AI inference engine is.","engine_id":"llamacpp-optimized"}'
```

> The first `/generate` loads the ~470 MB model into memory (~2–15 s depending
> on the host), then it stays cached for the process lifetime.

## Configuration

Set these in **Settings → Variables and secrets**:

| Variable | Example | Purpose |
|---|---|---|
| `ARMINFERX_CORS_ORIGINS` | `https://arminferx.vercel.app` | Browser origin of the deployed frontend (comma-separated for multiple) |

No model files are stored in this repo — `download_model.py` fetches and
verifies the Q4_K_M GGUF at build time.
