📊 Customer Churn Prediction 📉
A machine learning project to predict customer churn using Python and Scikit-learn.


https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python">
https://img.shields.io/badge/Status-Completed-green?style=for-the-badge">
https://img.shields.io/badge/License-MIT-purple?style=for-the-badge">

🚀 Overview
This project provides an end-to-end pipeline for predicting customer churn. By analyzing historical customer data, we train several classification models to identify customers at risk of leaving. This enables proactive customer retention strategies, which are crucial for business growth.

✨ Features
✅ End-to-End Pipeline: From data loading to model deployment.

✅ Exploratory Data Analysis (EDA): Generates insightful visualizations like correlation heatmaps and churn distribution plots.

✅ Multi-Model Training: Trains and evaluates Random Forest, Logistic Regression, and Decision Tree classifiers.

✅ Performance Evaluation: Automatically generates and saves Confusion Matrices and ROC Curves for each model.

✅ Modular Code: Helper functions are separated for better organization and reusability.

✅ Saved Models: Trained models are saved for easy reuse and integration.

🛠️ Technologies Used
Python

Pandas for data manipulation

Matplotlib & Seaborn for data visualization

Scikit-learn for machine learning models and metrics

Joblib for saving and loading models

📁 Directory Structure
├── 📂 Data/
│   └── customer_churn.csv      # Raw customer dataset
│
├── 🖼️ Image/
│   ├── churn_distribution.png    # Plot showing class distribution
│   ├── correlation_heatmap.png   # Heatmap of feature correlations
│   ├── *_confusion_matrix.png  # Confusion matrix for each model
│   └── *_roc_curve.png          # ROC curve for each model
│
├── 📦 Model/
│   ├── RandomForest.joblib       # Saved Random Forest model
│   ├── LogisticRegression.joblib # Saved Logistic Regression model
│   └── DecisionTree.joblib     # Saved Decision Tree model
│
├── 📜 main.py                     # Main script to run the pipeline
├── 📜 requirements.txt            # Required Python packages
└── 📜 utils.py                    # Helper functions for plotting and metrics
⚙️ How to Run
Follow these steps to set up and run the project locally.

1. Prerequisites
Make sure you have Python 3.8 or higher installed on your system.

2. Installation
Bash

# Clone the repository
git clone <your-repository-url>

# Navigate to the project directory
cd <your-repository-directory>

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
3. Execution
Run the main script to start the entire process:

Bash

python main.py
The script will automatically perform EDA, train the models, and save all outputs to the Image/ and Model/ directories.

🔧 Customization
This project is highly customizable:

Experiment with different models: Simply add or remove classifiers from the models dictionary in main.py.

Enhance functionality: Add new plotting functions or evaluation metrics to utils.py.

Use your own data: Replace customer_churn.csv with your dataset and adjust the data loading and preprocessing steps as needed.
