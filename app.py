import streamlit as st
import joblib
import numpy as np

st.title("ML Model Predictor")

model_choice = st.selectbox(
    "Choose a model",
    ["Logistic Regression (Iris - Classification)",
     "Decision Tree (Breast Cancer - Classification)",
     "Linear Regression (California Housing - Regression)"]
)

if model_choice == "Logistic Regression (Iris - Classification)":
    model = joblib.load("logistic_model.pkl")
    st.write("Enter 4 feature values (Sepal Length, Sepal Width, Petal Length, Petal Width):")
    f1 = st.number_input("Sepal Length", value=5.1)
    f2 = st.number_input("Sepal Width", value=3.5)
    f3 = st.number_input("Petal Length", value=1.4)
    f4 = st.number_input("Petal Width", value=0.2)
    features = np.array([[f1, f2, f3, f4]])
    classes = ["Setosa", "Versicolor", "Virginica"]
    if st.button("Predict"):
        pred = model.predict(features)[0]
        st.success(f"Predicted class: {classes[pred]}")

elif model_choice == "Decision Tree (Breast Cancer - Classification)":
    model = joblib.load("decisiontree_model.pkl")
    st.write("Enter values for the first 5 key features (others auto-filled with average):")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    avg_values = data.data.mean(axis=0)

    f1 = st.number_input("Mean Radius", value=float(avg_values[0]))
    f2 = st.number_input("Mean Texture", value=float(avg_values[1]))
    f3 = st.number_input("Mean Perimeter", value=float(avg_values[2]))
    f4 = st.number_input("Mean Area", value=float(avg_values[3]))
    f5 = st.number_input("Mean Smoothness", value=float(avg_values[4]))

    features = avg_values.copy()
    features[0], features[1], features[2], features[3], features[4] = f1, f2, f3, f4, f5
    features = features.reshape(1, -1)

    if st.button("Predict"):
        pred = model.predict(features)[0]
        label = "Malignant" if pred == 0 else "Benign"
        st.success(f"Prediction: {label}")

else:
    model = joblib.load("linear_model.pkl")
    st.write("Enter housing features:")
    medinc = st.number_input("Median Income (10k$)", value=3.5)
    house_age = st.number_input("House Age", value=25.0)
    ave_rooms = st.number_input("Average Rooms", value=5.0)
    ave_bedrms = st.number_input("Average Bedrooms", value=1.0)
    population = st.number_input("Population", value=1000.0)
    ave_occup = st.number_input("Average Occupancy", value=3.0)
    latitude = st.number_input("Latitude", value=34.0)
    longitude = st.number_input("Longitude", value=-118.0)

    features = np.array([[medinc, house_age, ave_rooms, ave_bedrms,
                           population, ave_occup, latitude, longitude]])

    if st.button("Predict"):
        pred = model.predict(features)[0]
        st.success(f"Predicted median house value: ${pred*100000:.2f}")
