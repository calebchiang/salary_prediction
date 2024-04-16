import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load('scaler.pkl')
model = load_model('salary_prediction_model.keras')

def main():
    print("Welcome to the Salary Predictor")
    x1 = float(input("Enter your years of experience: "))
    x1_scaled = scaler.transform(np.array([[x1]]))
    y_hat = model.predict(x1_scaled)
    print(f"Predicted Salary: ${y_hat[0][0]:,.2f}")


main()