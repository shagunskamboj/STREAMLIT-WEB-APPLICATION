import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load('career_predictor_model.pkl')
encoders = joblib.load('career_label_encoder.pkl')  # this should be a dict of encoders

# Title
st.title("🎓 Career Predictor App")
st.write("Fill in the student details below to predict a future career:")

# Selectbox inputs from encoders
gender = st.selectbox("Gender", encoders['gender'].classes_)
age = st.slider("Age", 18, 30)
gpa = st.slider("GPA", 2.0, 4.0, step=0.1)
major = st.selectbox("Major", encoders['major'].classes_)
interested_domain = st.selectbox("Interested Domain", encoders['interested_domain'].classes_)
projects = st.selectbox("Projects", encoders['projects'].classes_)
python = st.selectbox("Python Skill", encoders['python'].classes_)
sql = st.selectbox("SQL Skill", encoders['sql'].classes_)
java = st.selectbox("Java Skill", encoders['java'].classes_)

# Prepare input using encoders
input_data = np.array([
    encoders['gender'].transform([gender])[0],
    age,
    gpa,
    encoders['major'].transform([major])[0],
    encoders['interested_domain'].transform([interested_domain])[0],
    encoders['projects'].transform([projects])[0],
    encoders['python'].transform([python])[0],
    encoders['sql'].transform([sql])[0],
    encoders['java'].transform([java])[0]
]).reshape(1, -1)

# Predict and display
if st.button("Predict Career"):
    prediction = model.predict(input_data)
    predicted_label = encoders['future_career'].inverse_transform(prediction)
    st.success(f"Predicted Future Career: **{predicted_label[0]}**")
