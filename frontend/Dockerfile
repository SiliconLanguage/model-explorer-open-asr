# ── Stage 1: build the Vite / React app ──────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy source and build
COPY . .
RUN npm run build

# ── Stage 2: serve the built assets with Nginx ────────────────────────────────
FROM nginx:1.27-alpine

# Replace the default config with our custom one
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the compiled app
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
