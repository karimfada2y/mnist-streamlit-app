import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pandas as pd

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Classifier")

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():
    return {
        "SVM": joblib.load("svm_model.joblib"),
        "Logistic Regression": joblib.load("logistic_model.joblib"),
        "Decision Tree": joblib.load("tree_model.joblib"),
    }

models = load_models()
st.success("Models loaded successfully ✅")

# Column names exactly as the model was trained on
COLS = [f"{i//28 + 1}x{i%28 + 1}" for i in range(784)]

# ======================
# PREPROCESS IMAGE
# ======================
def preprocess_image(image):
    img = image.convert("L")        # grayscale
    img = img.resize((28, 28))

    img_array = np.array(img).astype(np.float32)

    # ✅ DO NOT invert — keep white background, dark digit
    # ✅ DO NOT normalize — model trained on 0–255 raw values

    st.image(img_array / 255.0, caption="Processed Image (as sent to model)", use_container_width=True)

    # Return as DataFrame with correct column names (required by Decision Tree)
    return pd.DataFrame(img_array.reshape(1, 784), columns=COLS)

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

# ======================
# PREDICTION
# ======================
if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    img_df = preprocess_image(image)

    results = []

    with col2:
        st.subheader("Results")

        for name, model in models.items():

            pred = model.predict(img_df)[0]

            # probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(img_df)[0]
            else:
                scores = model.decision_function(img_df)[0]
                scores = np.exp(scores - np.max(scores))  # stable softmax
                probs = scores / np.sum(scores)

            confidence = float(probs[pred])

            st.write(f"### {name}")
            st.metric("Prediction", pred, f"{confidence:.2%}")

            st.bar_chart(pd.DataFrame({
                "Digit": range(10),
                "Prob": probs
            }).set_index("Digit"))

            results.append([name, pred, f"{confidence:.2%}"])

    # ======================
    # SUMMARY
    # ======================
    df_results = pd.DataFrame(results, columns=["Model", "Prediction", "Confidence"])

    st.subheader("Comparison")
    st.table(df_results)