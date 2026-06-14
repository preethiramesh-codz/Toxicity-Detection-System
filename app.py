import streamlit as st
import joblib

# Page Settings
st.set_page_config(
    page_title="Toxicity Detection System",
    page_icon="🛡️",
    layout="centered"
)

# Load Model
model = joblib.load("model/toxicity_model.pkl")

# Title
st.title("🛡️ Toxicity Detection System")
st.write(
    "Detect whether a comment is Toxic or Non-Toxic using Machine Learning."
)

# Input Box
user_input = st.text_area(
    "Enter a comment:",
    height=150
)

# Prediction Button
if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:

        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.error("❌ Toxic Comment")
        else:
            st.success("✅ Non-Toxic Comment")

# Footer
st.markdown("---")
st.caption(
    "CodTech Internship Project | Toxicity Detection System"
)
