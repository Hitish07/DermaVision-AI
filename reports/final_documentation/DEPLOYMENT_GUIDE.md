# Deployment Guide

## Productionizing with Docker
The provided `docker-compose.yml` mounts volumes for persistent data (`skanscan.db`) and uploaded `figures`. 

For extreme production scale:
1. Swap the SQLite URL for PostgreSQL in `docker-compose.yml` (`DATABASE_URL=postgresql://user:pass@db:5432/skin`).
2. Implement a Redis Queue (Celery) instead of FastAPI `BackgroundTasks` to handle horizontal scaling of SHAP computation across multiple worker nodes.
