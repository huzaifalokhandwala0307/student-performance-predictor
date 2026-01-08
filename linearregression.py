from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Sample data: hours studied vs scores
data = pd.read_csv('student_study_stress_dataset.csv')

input_matrix = np.array(list(zip(data['Study_hours'], 
                                 data['Sleep_hours'], 
                                 data['Stress_level'], 
                                 data['Attendance'])))
scores = np.array(data['Marks'])
print("Input Matrix:\n", input_matrix)
# print("Data Head:\n", data.head())

# Reshape and scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(input_matrix)
scores = scores
X_train, X_test, y_train, y_test = train_test_split(x_scaled, scores, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# take user input and make prediction
Study_hours_exam = float(input("Enter study hours: "))
Sleep_hours_exam = float(input("Enter sleep hours: "))
Stress_level_exam = float(input("Enter stress level: "))
Attendance_exam = float(input("Enter attendance percentage: "))
input_data = np.array([[Study_hours_exam, Sleep_hours_exam, Stress_level_exam, Attendance_exam]])
input_scaled = scaler.transform(input_data)

# make prediction
predictions = model.predict(input_scaled)
print("Predicted marks:", round(predictions[0], 2))

# Evaluate the model
r2 = model.score(X_test, y_test)
print("R² score:", round(r2, 2))

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", round(mse, 2))

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", round(mae, 2))

# graph of actual vs predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=1)

# Perfect prediction line
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],
         linestyle='--',
         color='red')

plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.grid()
plt.legend(["Predictions", "Perfect Scores"])
plt.show() 


joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

