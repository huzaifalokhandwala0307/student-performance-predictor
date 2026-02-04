# 📊 Student Performance Predictor


A Machine Learning project that predicts **student marks** based on key academic and lifestyle factors such as **study hours, sleep hours, stress level, and attendance**.  
The project uses **Linear Regression** and provides both a **command-line prediction** and an **interactive Gradio web interface**.


---


## 🚀 Features


- 📈 Predict student marks using:
  - Study Hours
  - Sleep Hours
  - Stress Level
  - Attendance Percentage
- 🧠 Trained using **Linear Regression**
- 📏 Data normalization with **StandardScaler**
- 📊 Model evaluation using:
  - R² Score
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
- 📉 Visualization of **Actual vs Predicted Scores**
- 🌐 User-friendly **Gradio web app**
- 💾 Model and scaler saved using **Joblib**


---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Gradio  
- Joblib  

---

## 📂 Project Structure

Student-Performance-Predictor/
│
├── student_study_stress_dataset.csv
├── model.pkl
├── scaler.pkl
├── main.py
├── app.py
└── README.md

---

## 📊 Dataset Description

The dataset (`student_study_stress_dataset.csv`) contains the following columns:


| Column Name   | Description |
|--------------|-------------|
| Study_hours  | Hours spent studying per day |
| Sleep_hours  | Hours of sleep per day |
| Stress_level | Stress level (numeric scale) |
| Attendance   | Attendance percentage |
| Marks        | Final student marks |


---

## ⚙️ Installation

1. Clone the repository

git clone https://github.com/huzaifalokhandwala0307/student-performance-predictor.git

cd student-performance-predictor

2. Install dependencies

pip install numpy pandas scikit-learn matplotlib gradio joblib

---

## ▶️ Usage

### 🔹 Train the Model

Run the training script:

python main.py

This will:
- Train the model
- Evaluate performance
- Display a prediction graph
- Save `model.pkl` and `scaler.pkl`


---

### 🔹 Launch Gradio App

Run:

python app.py

Then open the local URL shown in the terminal to interact with the app.

---


## 📈 Model Evaluation Metrics


- **R² Score** – Measures goodness of fit
- **Mean Squared Error (MSE)** – Penalizes large errors
- **Mean Absolute Error (MAE)** – Average absolute prediction error

---

## 📉 Visualization

The project includes a scatter plot of:
- **Actual Marks vs Predicted Marks**
- Red dashed line represents **perfect prediction**

---

## 🎯 Example Prediction

**Input:**
- Study Hours: 6  
- Sleep Hours: 7  
- Stress Level: 3  
- Attendance: 85  

**Output:**

Predicted Marks: 78.42

---

## 🔮 Future Improvements

- Use advanced models (Random Forest, XGBoost)
- Add more features
- Deploy on cloud platforms
- Improve UI and validation

---

## 👨‍💻 Author


**Huzaifa**  
Machine Learning & Data Science Enthusiast  

---
