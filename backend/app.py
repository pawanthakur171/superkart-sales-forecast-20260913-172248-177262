
import io
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_model.joblib")
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURES = artifact["features"]
REFERENCE_YEAR = int(os.getenv("REFERENCE_YEAR", "2025"))

PERISHABLE_PRODUCT_TYPES = set(
    artifact.get(
        "perishable_product_types",
        ["Dairy", "Meat", "Seafood", "Fruits and Vegetables",
         "Breakfast", "Bread", "Frozen Foods"]
    )
)

app = Flask(__name__)


def add_engineered_features(dataframe):
    data = dataframe.copy()

    if "Product_Id" in data.columns:
        extracted = data["Product_Id"].astype(str).str.extract(
            r"^([A-Za-z]{2})", expand=False
        )
        data["Product_Id_char"] = (
            extracted.fillna(data["Product_Id"].astype(str).str[:2]).str.upper()
        )

    if "Store_Establishment_Year" in data.columns:
        data["Store_Age_Years"] = (
            REFERENCE_YEAR -
            pd.to_numeric(data["Store_Establishment_Year"], errors="coerce")
        ).clip(lower=0)

    if "Product_Type" in data.columns:
        data["Product_Type_Category"] = data["Product_Type"].map(
            lambda value: "Perishables" if value in PERISHABLE_PRODUCT_TYPES else "Non Perishables"
        )

    return data


def prepare_dataframe(dataframe):
    data = add_engineered_features(dataframe)

    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )

    data = data[FEATURES].copy()

    numeric = [
        "Product_Weight",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Age_Years",
    ]

    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise ValueError(
            "Missing/invalid values found in columns: " + ", ".join(bad)
        )

    return data


@app.get("/")
def home():
    return (
        "Welcome to the Backend — SuperKart App Project! "
        "Flask backend is working."
    )

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Welcome to the Backend — SuperKart App Project! Flask backend is working.",
        "model": artifact.get("model_name", "unknown"),
        "features": FEATURES
    })


@app.post("/v1/predict")
def predict():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        prepared = prepare_dataframe(pd.DataFrame([payload]))
        prediction = float(model.predict(prepared)[0])

        return jsonify({"prediction": prediction})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "Upload a CSV using form field 'file'"
            }), 400

        uploaded = request.files["file"]

        if not uploaded.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        raw = uploaded.read()
        input_df = pd.read_csv(io.BytesIO(raw))

        prepared = prepare_dataframe(input_df)
        predictions = model.predict(prepared)

        result = input_df.copy()
        result["Predicted_Product_Store_Sales_Total"] = predictions.astype(float)

        return jsonify({
            "predictions": result.to_dict(orient="records")
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
