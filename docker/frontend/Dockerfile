# =============================================================================
# ArmInferX — Frontend Dockerfile (production static build)
# =============================================================================
#
# Multi-stage build: a Node stage runs `npm run build` (Vite), then a minimal
# nginx image serves the static dist/ output. No Node runtime in production.
#
# The frontend calls the backend at an ABSOLUTE URL (API_URL in App.jsx, which
# reads VITE_API_URL at build time), so the build must know where the backend
# lives. Build from the repo root:
#
#   export VITE_API_URL=http://<SERVER_IP>:8000
#   docker compose build frontend          # or:
#   docker build -f docker/frontend/Dockerfile \
#       --build-arg VITE_API_URL=http://<SERVER_IP>:8000 \
#       -t arminferx-frontend .
#
# The backend must in turn allow this frontend's origin (ARMINFERX_CORS_ORIGINS
# env var, see docs/cloud-deploy.md).
# =============================================================================

FROM node:22-alpine AS build

# The API base URL baked into the JS bundle at build time.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

WORKDIR /app

# package.json first for layer caching; package-lock.json is gitignored, so
# `npm install` (not `npm ci`) is correct for fresh clones.
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# -- Runtime stage: static files only -----------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
