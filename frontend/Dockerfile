# =============================================================================
# Kael Trading Bot — Frontend Dockerfile
# =============================================================================
# Multi-stage build: compiles the Vite/React app in a Node builder, then
# serves the static assets with Nginx in a minimal production image.
# =============================================================================

# --- Build stage ---
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (layer-cached unless package-lock.json changes)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# --- Production stage ---
FROM nginx:alpine AS production

# Copy custom Nginx config that proxies /api to the backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the compiled static assets from the builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose the frontend HTTP port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
