Customer Churn Prediction Project
=================================

This project is designed to predict customer churn using various machine learning models. It includes exploratory data analysis (EDA), training and evaluation of multiple models, and visualization of results.

Directory Structure:
--------------------
├── Data/
│   └── customer_churn.csv      # Raw dataset
│
├── Image/
│   ├── churn_distribution.png  # Class balance plot
│   ├── correlation_heatmap.png # Correlation matrix
│   ├── *_confusion_matrix.png  # Confusion matrix per model
│   └── *_roc_curve.png         # ROC curves per model
│
├── Model/
│   ├── RandomForest.joblib     # Saved model
│   ├── LogisticRegression.joblib
│   └── DecisionTree.joblib
│
├── main.py                     # Main execution script
└── utils.py                    # Helper functions

How to Run:
-----------
1. Install dependencies:
   pip install -r requirements.txt

2. Execute the main script:
   python main.py

What It Does:
-------------
- Loads and preprocesses the dataset
- Performs exploratory data analysis (EDA)
- Trains multiple classification models:
  - Random Forest
  - Logistic Regression
  - Decision Tree
- Evaluates models using confusion matrices and ROC curves
- Saves visual outputs in the Image/ folder
- Saves trained models in the Model/ folder

Customization:
--------------
- Add or remove models in the `models` dictionary in `main.py`
- Modify `utils.py` to include more plots or metrics


