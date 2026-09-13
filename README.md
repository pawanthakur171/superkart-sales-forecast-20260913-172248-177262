# SuperKart Sales Forecast

## Project
SuperKart is a retail sales forecasting solution that predicts `Product_Store_Sales_Total` using historical product/store information.

## Repository structure
```text
README.md
backend/
  app.py
  Dockerfile
  requirements.txt
  superkart_model.joblib
  model_metadata.json
frontend/
  app.py
  Dockerfile
  requirements.txt
```

GitHub Codespaces provides a ready-to-use Docker environment; each service ships its own `Dockerfile` under `backend/` and `frontend/`.

## Backend
Flask API: port 7860
- `GET /health`
- `POST /v1/predict`
- `POST /v1/predictbatch`

The root endpoint (`GET /`) returns an explicit SuperKart backend message so that opening the forwarded backend port directly confirms that the Flask container is working.
The health endpoint (`GET /health`) returns JSON with status, model name and feature contract.

## Frontend
Streamlit UI: port 8501
- Single prediction form
- Drag-and-drop CSV upload for batch prediction
- Downloadable prediction CSV

## Runtime dependencies
The backend installs only what it actually needs at runtime:
- Flask for the REST API
- pandas for input validation and batch CSV handling
- scikit-learn because the serialized preprocessing/model pipeline contains sklearn objects
- joblib because the backend loads `superkart_model.joblib`
- only the external estimator package required by the selected final model, if applicable

The frontend installs only:
- Streamlit for the web UI
- requests for HTTP calls to Flask
- pandas for CSV display and prediction-result tables

The frontend does **not** load the model, so it does not install joblib, scikit-learn, XGBoost, CatBoost or LightGBM.
No Hugging Face packages are used because deployment is exclusively through GitHub Codespaces.

## Docker commands
Run these from the repository root in Codespaces.

Do **not** use `BACKEND_URL=http://backend:7860`. On GitHub Codespaces, container-to-container TCP to the `backend` hostname times out even when both containers share `rppapp-network`. The working path is the published host port via `host.docker.internal`.

```bash
docker rm -f frontend backend 2>/dev/null || true
docker network rm rppapp-network 2>/dev/null || true
docker network create rppapp-network
docker build -t backend ./backend
docker build -t frontend ./frontend
docker run -d --name backend --network rppapp-network -p 7860:7860 --restart unless-stopped backend
# wait until Flask answers on loopback (python:3.11-slim has no curl)
for i in $(seq 1 20); do docker exec backend python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=2)" >/dev/null 2>&1 && echo "backend healthy" && break || (echo "waiting for backend to be healthy... ($i)"; sleep 3); done
docker rm -f frontend
docker run -d --name frontend   --network rppapp-network   --add-host=host.docker.internal:host-gateway   -p 8501:8501   -e BACKEND_URL=http://host.docker.internal:7860   --restart unless-stopped   frontend
sleep 3
docker exec frontend python -c "import os,requests; print(os.environ['BACKEND_URL']); print(requests.get(os.environ['BACKEND_URL']+'/health', timeout=5).text)"
docker ps
# host-side checks (these work; http://backend:7860 from the Codespace shell will NOT resolve)
for i in $(seq 1 10); do curl -sf http://localhost:7860/health >/dev/null && echo "backend is up" && break || (echo "waiting for backend... ($i)"; sleep 3); done
curl http://localhost:7860/
for s in $(seq 10 -1 1); do printf "  frontend starting in %2ds " "$s"; sleep 1; done; printf "  frontend wait complete       
"
for i in $(seq 1 10); do curl -sf http://localhost:8501/ >/dev/null && echo "frontend is up" && break || (echo "waiting for frontend... ($i)"; sleep 3); done
# print BOTH forwarded application URLs
if [ -n "$CODESPACE_NAME" ]; then D="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"; echo "Backend  URL: https://${CODESPACE_NAME}-7860.${D}"; echo "Frontend URL: https://${CODESPACE_NAME}-8501.${D}"; fi
```

If the backend container is already healthy and only Streamlit cannot reach it, recreate **only** the frontend:

```bash
docker rm -f frontend
docker run -d --name frontend   --network rppapp-network   --add-host=host.docker.internal:host-gateway   -p 8501:8501   -e BACKEND_URL=http://host.docker.internal:7860   --restart unless-stopped   frontend
sleep 3
docker exec frontend python -c "import os,requests; print(os.environ['BACKEND_URL']); print(requests.get(os.environ['BACKEND_URL']+'/health', timeout=5).text)"
```

## Model cache
- Cache key: `df483c976c40e0132ae0c799840a5d9ed28e0f663504f88230f63c1fa331de3c`
- Training-data SHA256: `b12ec41668e69d71d64ca96f2be78a011087f2a9cac6aba6e5cc920544b21ab2`
- Selected model: `CatBoost`
- Reused on this run: `True`

## Model performance
The notebook compares the required regressors, tunes the strongest candidates, selects the final model, serializes the complete preprocessing + model pipeline, reloads it, and tests it on the held-out test set.

Reference year: `2025`
