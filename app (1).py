import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calorie Predictor")
st.write(
    "Enter nutritional information to estimate calories "
    "using a trained Decision Tree model."
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "calorie_model.pkl"
)

try:
    with open(MODEL_PATH, "rb") as file:
        bundle = pickle.load(file)

    model = bundle["model"]
    features = bundle["features"]

except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.subheader("Nutrition Information")

fat = st.number_input("Fat (g)", min_value=0.0, value=10.0, step=1.0)
carb = st.number_input("Carbohydrates (g)", min_value=0.0, value=40.0, step=1.0)
fiber = st.number_input("Fiber (g)", min_value=0.0, value=3.0, step=1.0)
protein = st.number_input("Protein (g)", min_value=0.0, value=8.0, step=1.0)

values = {
    "fat": fat,
    "carb": carb,
    "fiber": fiber,
    "protein": protein
}

if st.button("Predict Calories", type="primary"):
    try:
        input_data = pd.DataFrame(
            [[values[col] for col in features]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Calories: {prediction:.0f} kcal")
        st.caption(
            "This is an estimate from a machine learning model, "
            "not an official Starbucks nutrition value."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
