import streamlit as st
import joblib
import numpy as np

# Load models
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "MLP Classifier": joblib.load("mlp_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),
    "SVM Classifier": joblib.load("svm_model.pkl")
    
}

st.title("🚢 Titanic Survival Prediction")
st.write("Select a model and enter passenger details to predict survival chance.")

# Sidebar - model selection
model_name = st.selectbox("Choose ML Model", list(models.keys()))
model = models[model_name]

# Passenger input form
pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Port of Embarkation", ["Q", "S", "C"])

# Preprocess inputs
sex = 0 if sex == "Male" else 1
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Input array
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 Survived!")
    else:
        st.error("💀 Did not survive.")
