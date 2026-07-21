import gradio as gr
import joblib
import pandas as pd
import os

# Model path updated for root directory
model_path = "model.pkl" 
model = joblib.load(model_path)

def predict_churn(credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    input_data = pd.DataFrame([[
        credit_score, age, tenure, balance, num_of_products, 
        has_cr_card, is_active_member, estimated_salary
    ]], columns=[
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ])
    
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]
    
    result = "Churn (Customer will leave)" if prediction[0] == 1 else "Not Churn (Customer will stay)"
    return f"{result} (Probability: {proba:.2f})"

demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Credit Score", value=600),
        gr.Number(label="Age", value=40),
        gr.Number(label="Tenure", value=3),
        gr.Number(label="Balance", value=60000),
        gr.Number(label="Num Of Products", value=2),
        gr.Dropdown([0, 1], label="Has Credit Card (0 or 1)", value=1),
        gr.Dropdown([0, 1], label="Is Active Member (0 or 1)", value=1),
        gr.Number(label="Estimated Salary", value=50000)
    ],
    outputs="text",
    title="Customer Churn Prediction MLOps",
    description="Enter customer details to predict churn."
)

if __name__ == "__main__":
    demo.launch()