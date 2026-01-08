import gradio as gr
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_marks(study_hours, sleep_hours, stress_level, attendance):
    input_data = np.array([[study_hours, sleep_hours, stress_level, attendance]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return round(prediction[0], 2)

interface = gr.Interface(
    fn=predict_marks,
    inputs=[
        gr.Number(label="Study Hours"),
        gr.Number(label="Sleep Hours"),
        gr.Number(label="Stress Level"),
        gr.Number(label="Attendance (%)")
    ],
    outputs=gr.Number(label="Predicted Marks"),
    title="Student Performance Predictor",
    description="Predicts student marks based on study habits, sleep, stress, and attendance."
)

interface.launch()
