# ============================================================================
# Universal Maritime Ingestion Pipeline (V5.1 Production Dockerfile)
# Multi-Stage, Production-Hardened Build for Onboard Edge IPCs
# ============================================================================

# --- STAGE 1: Builder ---
FROM python:3.10-slim-buster AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- STAGE 2: Production Runtime ---
FROM python:3.10-slim-buster AS runner

# Security: Create non-root system user and group
RUN groupadd -g 10001 marine && \
    useradd -u 10001 -g marine -s /bin/bash -m marineapp

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application codebase and vessel configurations
COPY seakeeping_core /app/seakeeping_core
COPY vessels /app/vessels

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VESSEL_CONFIG=/app/vessels/sol_progress.yaml \
    UDP_PORT=10110

# Change ownership to non-root user
RUN chown -R marineapp:marine /app

USER marineapp

# Expose NMEA 0183 UDP Broadcast Port
EXPOSE 10110/udp

# Health check to ensure python process is active
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import seakeeping_core.ingestion.config_loader" || exit 1

# Default Entrypoint: Launch Live Pipeline on UDP Port 10110
ENTRYPOINT ["python3", "-m", "seakeeping_core.ingestion.live_pipeline"]
CMD ["--config", "vessels/sol_progress.yaml", "--udp-port", "10110"]
