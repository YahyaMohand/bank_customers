#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score, )
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


# In[3]:


dataset = pd.read_csv("../data/Churn Modeling.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.info()


# In[7]:


dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)


# In[8
# In[10]:


encoder = LabelEncoder()
dataset["Geography"] = encoder.fit_transform(dataset["Geography"])
dataset["Gender"] = encoder.fit_transform(dataset["Gender"])


# In[11]:


dataset["Age"].value_counts().plot.bar(figsize=(20,6))



scaler = MinMaxScaler()
df_scaled = pd.DataFrame(data = dataset)
df_scaled[:] = scaler.fit_transform(dataset[:])


# In[20]:


df_scaled.head()


# In[21]:


X = df_scaled.drop("Exited", axis=1)
y = df_scaled["Exited"]


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)


# In[26]:


with mlflow.start_run():
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))
    mlflow.log_param("alpha", accuracy_score(y_test, y_pred))
    mlflow.log_param("l1_ratio", precision_score(y_test, y_pred))
    mlflow.log_metric("rmse", recall_score(y_test, y_pred))
    mlflow.log_metric("r2", f1_score(y_test, y_pred))

    remote_server_uri="https://dagshub.com/YahyaMohand/bank_customers.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
    if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            clf, "model", registered_model_name="clf"
        )
    else:
        mlflow.sklearn.log_model(clf, "model")


# In[ ]:




