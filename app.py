import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

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

# Service columns for feature engineering
service_cols = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

# Engineered features
engineered_numerical = ['AvgChargesPerMonth']
engineered_categorical = ['TenureCategory', 'TotalServicesSubscribed']

# Final feature lists
all_numerical = numerical_cols + engineered_numerical
all_categorical = categorical_cols + engineered_categorical

def categorize_churn(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.1:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def preprocess_input(data):
    # Create DataFrame from input data
    df = pd.DataFrame([data])
    
    # 1. Ensure all expected columns exist
    for col in service_cols:
        if col not in df.columns:
            df[col] = 'No'  # Default value for service columns
    
    # 2. Handle numerical features
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Feature engineering
    df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    df['TotalServicesSubscribed'] = df[service_cols].apply(
        lambda x: sum(val == 'Yes' for val in x), 
        axis=1
    ).astype(str)  # Treat as categorical
    
    df['TenureCategory'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 48, 72],
        labels=['Short-Term', 'Mid-Term', 'Long-Term'],
        include_lowest=True
    )
    df['TotalCharges'] = np.sqrt(df['TotalCharges'])  # Apply transformation
    
    # 4. Encode categorical features (including engineered ones)
    for col in all_categorical:
        le = label_encoders.get(col)
        if le:
            df[col] = df[col].astype(str)
            # Handle unseen categories
            mask = ~df[col].isin(le.classes_)
            if mask.any():
                most_common = le.classes_[0]
                df.loc[mask, col] = most_common
            df[col] = le.transform(df[col])
    
    # 5. Scale numerical features
    df[all_numerical] = scaler.transform(df[all_numerical])
    
    # 6. Ensure feature order matches training
    try:
        return df[model.feature_names_in_]
    except AttributeError:
        return df[all_numerical + all_categorical]

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
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
        prob = model.predict_proba(X)[0][1]  # Probability of churn (class 1)
        prediction = categorize_churn(prob)
        
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    # Verify feature consistency
    print("Model expects features:", model.feature_names_in_)
    app.run(debug=True)