from utils import load_data, feature_engineering, preprocess_data, split_data, evaluate_model, perform_eda, evaluate_and_plot_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

def main():
    data_path = "Data/customer_churn.csv"  
    df = load_data(data_path)

    perform_eda(df)

    df = feature_engineering(df)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_and_plot_model(model, X_test, y_test, name)
        save_model(model, name)

if __name__ == "__main__":
    main()
