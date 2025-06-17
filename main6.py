import streamlit as st
import numpy as np
import joblib

# Load models and preprocessing tools
baseline_model = joblib.load("baseline_classifier.pkl")
tuned_model = joblib.load("tuned_classifier.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Feature order and updated min/max ranges from your dataset
original_feature_ranges = {
    "Density": (0.995, 1.1089, "float"),
    "Age": (22, 81, "int"),
    "Weight": (118.5, 363.15, "float"),
    "Height": (29.5, 77.75, "float"),
    "Neck": (31.1, 51.2, "float"),
    "Chest": (79.3, 136.2, "float"),
    "Abdomen": (69.4, 148.1, "float"),
    "Hip": (85.0, 147.7, "float"),
    "Thigh": (47.2, 87.3, "float"),
    "Knee": (33.0, 49.1, "float"),
    "Ankle": (19.1, 33.9, "float"),
    "Biceps": (24.8, 45.0, "float"),
    "Forearm": (21.0, 34.9, "float"),
    "Wrist": (15.8, 21.4, "float"),
}

feature_ranges = {}

for feature, (min_val, max_val, dtype) in original_feature_ranges.items():
    if feature == "Age":
        feature_ranges[feature] = (1, 100, "int")
    else:
        range_extension = 0.3 * (max_val - min_val)
        new_min = min_val - range_extension
        new_max = max_val + range_extension
        feature_ranges[feature] = (new_min, new_max, dtype)

feature_names = list(feature_ranges.keys())

# Streamlit page config
st.set_page_config(page_title="Body Category Classifier", layout="centered")

# App Header
st.markdown("""
    <h1 style='text-align: center; color: #3b5998;'>🏋️‍♂️ Body Type Classifier</h1>
    <p style='text-align: center;'>Predict your body composition category using physical measurements.</p>
    <hr style="border: 1px solid #eee;">
""", unsafe_allow_html=True)

# Model selector
model_option = st.selectbox("🔍 Choose a model to use:", ["Select a model", "Baseline Model", "Fine-Tuned Model"])
model = baseline_model if model_option == "Baseline Model" else tuned_model if model_option == "Fine-Tuned Model" else None

if model:
    st.markdown("### 🛠️ Input your measurements")

    input_values = []
    cols = st.columns(2)

    for i, (feature, (min_val, max_val, dtype)) in enumerate(feature_ranges.items()):
        with cols[i % 2]:
            if feature == "Density":
                value = st.slider(
                    f"{feature} ({min_val:.3f}–{max_val:.3f})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=round((min_val + max_val) / 2, 3),
                    step=0.001
                )
            elif dtype == "int":
                value = st.slider(
                    f"{feature} ({int(min_val)}–{int(max_val)})",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int((min_val + max_val) / 2),
                    step=1
                )
            else:
                value = st.slider(
                    f"{feature} ({min_val:.2f}–{max_val:.2f})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=round((min_val + max_val) / 2, 2),
                    step=0.1
                )
            input_values.append(value)

    # Predict on button click
    if st.button("🔍 Predict Category"):
        scaled_input = scaler.transform([input_values])
        pred_encoded = model.predict(scaled_input)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.markdown("### 🧠 Predicted Category:")
        st.success(f"🎯 **{pred_label}**")

        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; font-size: 14px; color: grey;'>Developed by <strong>Laxman</strong>, <strong>Nirjala</strong> & <strong>Sandesh</strong></p>",
            unsafe_allow_html=True
        )
else:
    st.info("👈 Please select a model to continue.")
