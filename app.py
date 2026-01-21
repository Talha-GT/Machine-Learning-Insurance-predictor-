import gradio as gr
import pandas as pd
import pickle

# Load the trained model
with open("gb_model.pkl", "rb") as f:
    best_model = pickle.load(f)

def predict(age, sex, bmi, children, smoker, region):
    data = {
        'age':[age],
        'sex':[1 if sex=="male" else 0],
        'bmi':[bmi],
        'children':[children],
        'smoker':[1 if smoker=="yes" else 0],
        'region_northwest':[1 if region=="northwest" else 0],
        'region_southeast':[1 if region=="southeast" else 0],
        'region_southwest':[1 if region=="southwest" else 0],
    }
    df_input = pd.DataFrame(data)
    return int(best_model.predict(df_input)[0])

gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["male","female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Radio(["yes","no"], label="Smoker"),
        gr.Dropdown(["northwest","southeast","southwest","northeast"], label="Region")
    ],
    outputs="number",
    title="Insurance Cost Predictor"
).launch()
