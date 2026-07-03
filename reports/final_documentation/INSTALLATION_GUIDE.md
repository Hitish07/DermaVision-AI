# Installation Guide

## Requirements
- Docker >= 24.0
- Docker Compose >= 2.0

## Steps
1. Clone the repository.
2. Ensure port `80` and `8000` are free on the host machine.
3. Run: `docker-compose up --build -d`
4. Access the web interface at `http://localhost`.

## Local Development
- **Backend**: `pip install -r requirements.txt && uvicorn api.main:app`
- **Frontend**: `cd frontend && npm install && npm run dev`
