import streamlit as st
import joblib

# Load Model
model = joblib.load("model/toxicity_model.pkl")

# Page Settings
st.set_page_config(
    page_title="Toxicity Detection System",
    page_icon="🛡️",
    layout="centered"
)

# Title
st.title("🛡️ Toxicity Detection System")
st.write("Detect whether a comment is Toxic or Non-Toxic using Machine Learning.")

# Text Input
user_input = st.text_area(
    "Enter a comment:",
    height=150,
    placeholder="Type a comment here..."
)

# Prediction Button
if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")

# Footer
st.markdown("---")
st.caption("CodTech Internship Project | Toxicity Detection System")