import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


remote_server_uri = "https://dagshub.com/YahyaMohand/bank_customers.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

# Load the dataset
dataset = pd.read_csv("../data/Churn Modeling.csv")

# Drop unnecessary columns
dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
dataset["Geography"] = encoder.fit_transform(dataset["Geography"])
dataset["Gender"] = encoder.fit_transform(dataset["Gender"])

# Scale features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(data=dataset)
df_scaled[:] = scaler.fit_transform(dataset[:])

# Define features and target variable
X = df_scaled.drop("Exited", axis=1)
y = df_scaled["Exited"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2 * n for n in range(1, 10)],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

clf = RandomForestClassifier(criterion='gini', max_depth=18, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, random_state=200)
clf_cv = GridSearchCV(clf, parameters, cv=10)
clf_cv.fit(X_train, y_train)

# Start an MLflow run
with mlflow.start_run():

    # Log hyperparameters from GridSearchCV
    mlflow.log_params(clf_cv.best_params_)

    # Initialize and train the RandomForestClassifier with tuned hyperparameters
    clf = RandomForestClassifier(**clf_cv.best_params_, random_state=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate the model and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("average_precision", average_precision)

    # Set the remote MLflow tracking server URI
    remote_server_uri = "https://dagshub.com/YahyaMohand/bank_customers.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)