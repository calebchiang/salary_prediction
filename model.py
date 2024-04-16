from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("salary_data.csv")
X = df[['YearsExperience']]
y = df['Salary']

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]) # output layer is just a linear regression model
])

def main():
    optimizer = tf.optimizers.SGD(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=len(X_train))
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    model.save("salary_prediction_model.keras")
    joblib.dump(scaler, 'scaler.pkl')

main()
