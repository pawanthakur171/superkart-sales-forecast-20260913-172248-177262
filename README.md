# SuperKart Sales Forecast

SuperKart is a retail sales forecasting solution that predicts `Product_Store_Sales_Total`.

## Repository structure
```text
README.md
backend/
frontend/
```

The final deployment files are published into `backend/` and `frontend/` after model training/serialization.

The notebook prints the complete Docker command sequence required to build the backend and frontend images, create the shared `rppapp-network`, run both containers, verify the backend health endpoint, and use the forwarded ports.
