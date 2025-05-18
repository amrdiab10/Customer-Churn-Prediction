import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
# ------------------------- #
# 1. Load Data
# ------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

# ------------------------- #
# 2. Feature Engineering
# ------------------------- #
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['tenure'].replace(0, 1, inplace=True)  
    df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure']
    df['TenureCategory'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 48, np.inf],
        labels=['Short-Term', 'Mid-Term', 'Long-Term']
    )

    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['TotalServicesSubscribed'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

    return df

# ------------------------- #
# 3. Preprocess Features
# ------------------------- #
def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    df.drop(['customerID', 'Unnamed: 0'], axis=1, errors='ignore', inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    if 'TenureCategory' in X.columns:
        X['TenureCategory'] = X['TenureCategory'].astype(str)

    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y

def perform_eda(df: pd.DataFrame, image_dir='Image'):
    os.makedirs(image_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.savefig(os.path.join(image_dir, 'churn_distribution.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(image_dir, 'correlation_heatmap.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='tenure', bins=20, kde=True)
    plt.title('Tenure Distribution')
    plt.savefig(os.path.join(image_dir, 'tenure_distribution.png'))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title('Monthly Charges by Churn')
    plt.savefig(os.path.join(image_dir, 'monthly_charges_by_churn.png'))
    plt.close()

    print(f"[INFO] EDA visuals saved in '{image_dir}/'")



# ------------------------- #
# 4. Split Data
# ------------------------- #
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# ------------------------- #
# 5. Evaluate Model
# ------------------------- #
def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

def evaluate_and_plot_model(model, X_test, y_test, model_name, image_dir='Image'):
    os.makedirs(image_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(os.path.join(image_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f'{model_name} - ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(image_dir, f'{model_name}_roc_curve.png'))
        plt.close()

    print(f"[INFO] {model_name} - Evaluation plots saved.")


def save_model(model, model_name, model_dir='Model'):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{model_name}.joblib')
    joblib.dump(model, path)
    print(f"[INFO] {model_name} saved to {path}")    
