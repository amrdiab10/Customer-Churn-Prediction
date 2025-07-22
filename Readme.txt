Customer Churn Prediction
Overview
This project aims to predict customer churn using machine learning. By analyzing customer data, we build and evaluate several classification models to identify customers who are likely to stop using a service. This allows businesses to take proactive measures to retain valuable customers. The project includes all steps from data loading and exploratory data analysis (EDA) to model training, evaluation, and storage.

Directory Structure
├── Data/
│   └── customer_churn.csv      # Raw customer dataset
│
├── Image/
│   ├── churn_distribution.png    # Plot showing class distribution (Churn vs. No Churn)
│   ├── correlation_heatmap.png   # Heatmap of feature correlations
│   ├── *_confusion_matrix.png  # Confusion matrix for each model
│   └── *_roc_curve.png          # ROC curve for each model
│
├── Model/
│   ├── RandomForest.joblib       # Saved Random Forest model
│   ├── LogisticRegression.joblib # Saved Logistic Regression model
│   └── DecisionTree.joblib     # Saved Decision Tree model
│
├── main.py                     # Main script to run the entire pipeline
├── requirements.txt            # List of required Python packages
└── utils.py                    # Helper functions for plotting and metrics
How to Run
Follow these steps to get the project up and running on your local machine.

Prerequisites
Ensure you have Python 3.8 or higher installed.

Installation
Clone the repository:

Bash

git clone <your-repository-url>
cd <your-repository-directory>
Create a virtual environment (optional but recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:

Bash

pip install -r requirements.txt
Execution
Run the main script to execute the entire data processing, training, and evaluation pipeline:

Bash

python main.py
What It Does
The main.py script automates the following processes:

Data Loading: Loads the customer_churn.csv dataset from the Data/ directory.

Exploratory Data Analysis (EDA):

Generates and saves a class balance plot (churn_distribution.png).

Computes and visualizes a correlation matrix (correlation_heatmap.png).

Model Training:

Trains multiple classification models as defined in the script (default: Random Forest, Logistic Regression, Decision Tree).

Model Evaluation:

For each model, it computes and saves a confusion matrix and ROC curve in the Image/ folder.

Model Saving:

Saves the trained models as .joblib files in the Model/ directory for future use.

Customization
You can easily customize this project:

Add or Remove Models: Modify the models dictionary in main.py to experiment with different classifiers from scikit-learn.

Extend Functionality: Add new plotting functions or evaluation metrics to utils.py.

Change the Dataset: Replace customer_churn.csv with your own dataset. You may need to adjust feature engineering steps accordingly.
