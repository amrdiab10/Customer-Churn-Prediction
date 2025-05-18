import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# Load trained model and preprocessing tools
model = joblib.load('Model/stacking_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

app = Flask(__name__)

# Base columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Engineered features
engineered_numerical = ['AvgChargesPerMonth', 'TotalServicesSubscribed']
engineered_categorical = ['TenureCategory']
service_cols = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

# Full list of features
all_numerical = numerical_cols + engineered_numerical
all_categorical = categorical_cols + engineered_categorical

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Drop ID if present
    df = df.drop(columns=['customerID'], errors='ignore')

    # Convert numeric columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Manual Feature Engineering
    df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    df['TenureCategory'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 48, 72],
        labels=['Short-Term', 'Mid-Term', 'Long-Term'],
        include_lowest=True
    )

    df['TotalServicesSubscribed'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

    # Encode categorical features
    for col in all_categorical:
        le = label_encoders.get(col)
        if le:
            df[col] = df[col].astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            raise ValueError(f"LabelEncoder for '{col}' is missing.")

    # Scale numerical features
    df[all_numerical] = scaler.transform(df[all_numerical])

    # Return full processed DataFrame
    return df[all_numerical + all_categorical]

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'customerID': request.form.get('customerID'),
            'gender': request.form.get('gender'),
            'SeniorCitizen': request.form.get('SeniorCitizen'),
            'Partner': request.form.get('Partner'),
            'Dependents': request.form.get('Dependents'),
            'tenure': request.form.get('tenure'),
            'PhoneService': request.form.get('PhoneService'),
            'MultipleLines': request.form.get('MultipleLines'),
            'InternetService': request.form.get('InternetService'),
            'OnlineSecurity': request.form.get('OnlineSecurity'),
            'OnlineBackup': request.form.get('OnlineBackup'),
            'DeviceProtection': request.form.get('DeviceProtection'),
            'TechSupport': request.form.get('TechSupport'),
            'StreamingTV': request.form.get('StreamingTV'),
            'StreamingMovies': request.form.get('StreamingMovies'),
            'Contract': request.form.get('Contract'),
            'PaperlessBilling': request.form.get('PaperlessBilling'),
            'PaymentMethod': request.form.get('PaymentMethod'),
            'MonthlyCharges': request.form.get('MonthlyCharges'),
            'TotalCharges': request.form.get('TotalCharges'),
        }

        # Preprocess and predict
        X = preprocess_input(data)
        prediction = model.predict(X)[0]
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
