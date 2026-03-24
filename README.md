# AFVE (Agentic Factual Verification Engine)

This repo contains:
- FastAPI backend (Python) in the repo root
- Vite + React frontend in `Frontend/`

## Run locally (development)

### 1) Backend (FastAPI)

In one terminal:

- Create/activate your Python env and install deps:
  - `python -m pip install -r requirements.txt`
- Start the API server:
  - `python -m uvicorn main:app --reload --port 8000`

Backend will be available at:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/verify`

### 2) Frontend (Vite)

In a second terminal:

- `cd Frontend`
- `npm install`
- `npm run dev`

Frontend will be available at:
- `http://localhost:8080/`

### How frontend talks to backend

During development, Vite proxies `http://localhost:8080/api/*` → `http://127.0.0.1:8000/*`.
So the UI calls `/api/verify` (no CORS hassle).

Optional env vars live in `Frontend/.env.example`.
