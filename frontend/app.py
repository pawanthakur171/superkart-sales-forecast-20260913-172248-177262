
import os
import time
import requests
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://host.docker.internal:7860").rstrip("/")

# Separate connect vs read timeouts: fail fast on a network/DNS problem (connect),
# but allow the model time to respond (read).
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="\U0001F6D2", layout="centered")

st.title("\U0001F6D2 Welcome to the SuperKart App Project")
st.caption("Streamlit frontend -> Flask backend -> serialized ML pipeline")


def check_backend():
    """Return (ok, detail). Never raises."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=(CONNECT_TIMEOUT, 15))
        if r.status_code == 200:
            data = r.json()
            return True, data.get("model", "unknown")
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:
        return False, str(exc)


# ---- Startup connectivity banner: tells the user immediately if the backend is reachable ----
with st.spinner(f"Checking backend at {BACKEND_URL} ..."):
    ok, detail = check_backend()

if ok:
    st.success(f"Backend connected at {BACKEND_URL} (model: {detail})")
else:
    st.error(
        "Cannot reach the backend from the frontend container.\n\n"
        f"- Tried: {BACKEND_URL}/health\n"
        f"- Error: {detail}\n\n"
        "This is almost always a Codespaces Docker ICC issue, not a model problem. "
        "Direct http://backend:7860 is blocked; use the published host port "
        "(host.docker.internal:7860) and recreate the frontend as shown:"
    )
    st.code(
        "docker network inspect rppapp-network --format '{{range .Containers}}{{.Name}} {{end}}'\n"
        "docker exec frontend python -c \"import requests; print(requests.get('http://host.docker.internal:7860/health', timeout=5).text)\"\n"
        "# Codespaces blocks container-to-container TCP to http://backend:7860.\n"
        "# Recreate the frontend through the published host port:\n"
        "docker rm -f frontend 2>/dev/null || true\n"
        "docker run -d --name frontend --network rppapp-network --add-host=host.docker.internal:host-gateway -p 8501:8501 -e BACKEND_URL=http://host.docker.internal:7860 --restart unless-stopped frontend",
        language="bash",
    )
    st.info("Fix the network above, then click 'Rerun' (top-right) or refresh this page.")

st.divider()
st.subheader("Online prediction")

with st.form("prediction_form"):
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    Product_Id_char = st.text_input("Product ID prefix", value="FD", max_chars=2)
    Store_Age_Years = st.number_input("Store Age (years)", min_value=0, value=16)
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
    submitted = st.form_submit_button("Predict Sales")

if submitted:
    payload = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char.upper(),
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category,
    }
    # Live, transparent processing UI with a spinner and step-by-step status.
    with st.status("Processing your request in the backend...", expanded=True) as status:
        try:
            st.write(f"Connecting to backend at {BACKEND_URL} ...")
            t0 = time.time()
            st.write("Sending features to POST /v1/predict ...")
            response = requests.post(
                f"{BACKEND_URL}/v1/predict",
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            st.write(f"Backend responded with HTTP {response.status_code} in {time.time()-t0:.2f}s. Parsing...")
            response.raise_for_status()
            prediction = response.json()["prediction"]
            status.update(label="Prediction complete", state="complete", expanded=False)
            st.success(f"Predicted sales: {prediction:,.2f}")
        except requests.exceptions.ConnectTimeout as exc:
            status.update(label="Backend unreachable (connection timed out)", state="error")
            st.error(
                "The request timed out while trying to CONNECT to the backend. "
                "The backend is likely not reachable on the Docker network "
                f"({BACKEND_URL}). See the network fix commands at the top of the page."
            )
            st.exception(exc)
        except requests.exceptions.ReadTimeout as exc:
            status.update(label="Backend timed out while processing", state="error")
            st.warning("The backend accepted the request but took too long to respond. "
                       "Please wait a moment and try again.")
            st.exception(exc)
        except Exception as exc:
            status.update(label="Prediction failed", state="error")
            st.error("The backend returned an error. Full details below:")
            st.exception(exc)
            try:
                st.write("Backend response body:")
                st.code(response.text)
            except Exception:
                pass

st.divider()
st.subheader("Batch Prediction")
st.write("Drag and drop a CSV file below, or click Browse files. The uploaded file is sent to the Flask backend for batch inference.")

uploaded = st.file_uploader("Drag and drop Batch_Data_SuperKart.csv here", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    st.dataframe(batch_df.head(20), use_container_width=True)
    if st.button("Run Batch Prediction"):
        with st.status("Running batch prediction in the backend...", expanded=True) as status:
            try:
                st.write(f"Uploading {uploaded.name} ({len(batch_df)} rows) to POST /v1/predictbatch ...")
                prog = st.progress(0.3, text="Backend is scoring the rows...")
                t0 = time.time()
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
                prog.progress(0.7, text=f"Backend responded (HTTP {response.status_code}). Parsing predictions...")
                response.raise_for_status()
                result_df = pd.DataFrame(response.json()["predictions"])
                prog.progress(1.0, text="Done")
                status.update(label=f"Batch prediction complete in {time.time()-t0:.2f}s", state="complete", expanded=False)
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
            except requests.exceptions.ConnectTimeout as exc:
                status.update(label="Backend unreachable (connection timed out)", state="error")
                st.error("Could not connect to the backend. See the network fix commands at the top of the page.")
                st.exception(exc)
            except Exception as exc:
                status.update(label="Batch prediction failed", state="error")
                st.error("The batch request failed. Full details below:")
                st.exception(exc)
                try:
                    st.write("Backend response body:")
                    st.code(response.text)
                except Exception:
                    pass

