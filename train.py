import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    """
    Downloads the telco churn dataset, preprocesses features, 
    trains a Random Forest classifier, and serializes the model.
    """
    os.makedirs('model', exist_ok=True)

    logging.info("Downloading dataset...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    logging.info("Preprocessing data...")
    df = df.drop('customerID', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.4f}")

    model_path = 'model/churn_rf_model.pkl'
    joblib.dump(model, model_path)
    logging.info(f"Model serialized at: {model_path}")

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'model/model_features.pkl')
    logging.info("Feature names serialized at: model/model_features.pkl")
    logging.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_model()
