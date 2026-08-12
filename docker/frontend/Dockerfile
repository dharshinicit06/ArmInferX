# =============================================================================
# ArmInferX — Frontend Dockerfile (placeholder)
# =============================================================================
#
# STATUS: Not configured yet. This file currently contains only planning
# comments. It will be filled in during the Docker setup phase (Phase 6 of
# the roadmap).
#
# PURPOSE
# -------
# Builds a Docker image for the React + Vite frontend located in /frontend.
# Two common strategies exist — pick one during the setup phase:
#
#   A) DEV / STATIC  — build the app with `npm run build` and serve the
#      resulting static files (dist/) with a lightweight web server such as
#      nginx or caddy. Best for production: small image, no Node runtime.
#
#   B) MULTI-STAGE   — one stage installs Node and runs the Vite build; a
#      second stage copies dist/ into a minimal runtime image. Recommended,
#      as it keeps the final image small.
#
# WHAT WILL GO HERE LATER
# -----------------------
# 1. BASE IMAGE
#    node:22-alpine (or similar) for the build stage; nginx:alpine (or
#    caddy) for the static serving stage.
#
# 2. DEPENDENCIES
#    - Copy frontend/package.json (and package-lock.json when committed)
#      and run `npm ci` to install locked dependencies reproducibly.
#
# 3. BUILD
#    - Copy the frontend source and run `npm run build` to produce dist/.
#
# 4. RUNTIME
#    - Serve dist/ on port 80 with nginx (or the chosen server).
#    - If the SPA gains client-side routing later, the server must include
#      a fallback to index.html for unknown routes.
#
# EXAMPLE SKELETON (do not enable yet):
#
#   FROM node:22-alpine AS build
#   WORKDIR /app
#   COPY frontend/package.json ./
#   RUN npm ci
#   COPY frontend/ .
#   RUN npm run build
#
#   FROM nginx:alpine
#   COPY --from=build /app/dist /usr/share/nginx/html
#   EXPOSE 80
