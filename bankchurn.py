#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

import time


# In[2]:


df0 = pd.read_csv("Bank_Churn_Modelling.csv")

df0.head(10)


# In[3]:


df0.info()


# In[4]:


df0.describe()


# In[5]:


df0.shape


# In[6]:


df0.columns


# In[7]:


df0.isna().sum()


# In[8]:


df0.dropna(subset=['Geography', 'Age', 'HasCrCard', 'IsActiveMember'], axis=0, inplace=True)

print(df0.shape)


# In[9]:


df0.head()


# In[10]:


df0.size


# In[11]:


df0.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)


# In[12]:


df0.duplicated().sum()


# In[13]:


df0[df0.duplicated()].head(10)


# In[14]:


df1 = df0.drop_duplicates(keep='first')


# In[15]:


df1.head()


# In[16]:


# Determining the number of rows containing outliers
plt.figure(figsize=(5,1.5))
plt.title('Boxplot to detect outliers for CreditScore', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['CreditScore'])
plt.show()


# In[17]:


# Determining the number of rows containing outliers
plt.figure(figsize=(5,1.5))
plt.title('Boxplot to detect outliers for Age', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['Age'])
plt.show()


# In[18]:


# Determining the number of rows containing outliers
plt.figure(figsize=(5,1.5))
plt.title('Boxplot to detect outliers for Tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['Tenure'])
plt.show()


# In[19]:


# Numbers of people who Exited vs. stayed
print(df1['Exited'].value_counts())
print()

# Percentages of people who Exited vs. stayed
print(df1['Exited'].value_counts(normalize=True))


# In[ ]:





# ### Modeling Approach: Logistic Regression Model

# In[20]:


# Copying the dataframe
df1_enc = df1.copy()

# Encoding the 'Gender' column as an ordinal numeric category (Female: 0, Male: 1)
df1_enc['Gender'] = (
    df1_enc['Gender'].astype('category')
    .cat.set_categories(['Female', 'Male'])
    .cat.codes
)

# Dummy encoding the 'Geography' column
df1_enc = pd.get_dummies(df1_enc, columns=['Geography'], drop_first=False)

# Displaying the first 10 rows of the encoded dataframe
df1_enc.head(10)


# In[ ]:





# In[21]:


# heatmap to visualize how correlated viriables are
plt.figure(figsize=(6,4))
sns.heatmap(df1_enc[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']]
            .corr(), annot=True, cmap="crest")
plt.title('Heatmap for the dataset')
plt.show()


# In[22]:


# In the legend, the color purple (0) represents customers who stayed, while the color green (1) represents employees who left.
pd.crosstab(df1['Tenure'], df1['Exited']).plot(kind='bar', color='gm')
plt.title('Counts of Customers who Exited versus Stayed accross Department')
plt.ylabel('Customer Count')
plt.xlabel('Tenure')
plt.show()


# In[23]:


# Define features and target
X = df1_enc.drop('Exited', axis=1)
y = df1_enc['Exited']


# In[24]:


# Splitting the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


# In[25]:


log_clf = LogisticRegression(random_state=42, max_iter=500)
log_clf.fit(X_train, y_train)


# In[26]:


# Using the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)


# In[27]:


# Computing values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Creating display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,
                                  display_labels=log_clf.classes_)

# Plotting confusion matrix
log_disp.plot(values_format='', cmap='Oranges')

# Displaying plot
plt.show()


# True Negatives (TN = 1945): Model correctly predicted "No" for 1945 customers who actually did not churn.
# 
# False Positives (FP = 45): Model incorrectly predicted "Yes" (churn) for 45 customers who did not churn.
# 
# False Negatives (FN = 473): Model missed 473 actual churners, predicting they would stay (this is a critical miss depending on your business objective).
# 
# True Positives (TP = 36): Model correctly identified 36 churners.

# In[28]:


df1_enc['Exited'].value_counts(normalize=True)


# Exited = 0 → Customers who did not churn (stayed)
# → 79.6% of the customers
# 
# Stay = 1 → Customers who churned (left the company)
# → 20.4% of the customers

# In[29]:


# Creating classification report for logistic regression model
target_names = ['Predicted would Stay', 'Predicted would Exist']
print(classification_report(y_test, y_pred, target_names=target_names))


# The logistic regression model performs well in predicting customers who stay (98% recall) but poorly for those who exit (only 7% recall).

# ### Random Forest

# In[50]:


# Instantiating the mdoel
rf= RandomForestClassifier(random_state=0)

# Assigning a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3, 5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [2, 3, 4],
            }

# Assigning a dictionary of scoring metrics to capture
scoring = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'precision': 'precision'
}
# Instantiating GridSearch
rf1 = GridSearchCV(
    rf, 
    cv_params, 
    scoring=scoring, 
    cv=4, 
    refit='roc_auc'
)


# In[31]:


# ⏱️ Timing the fitting process
start_time = time.time()
rf1.fit(X_train, y_train)
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds")


# In[32]:


# Best AUC score
print("Best CV AUC Score:", rf1.best_score_)

# Best parameters
print("Best Parameters:", rf1.best_params_)


# In[33]:


def get_scores(model_name: str, 
               model, 
               X_test_data, 
               y_test_data):
    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table


# In[ ]:





# In[34]:


# Getting predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores


# The tuned Random Forest model achieved an AUC of 0.85 during cross-validation, indicating strong discriminatory power.
# 
# On the test set, it performed with 86% accuracy, 0.73 precision, 0.50 recall, and an AUC of 0.72, showing improved balance over logistic regression.

# In[35]:


# Generating array of values for confusion matrix
preds = rf1.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf1.classes_)

# Plotting confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf1.classes_)
disp.plot(values_format='');


# True Negatives (TN = 1896): The model correctly predicted 1896 customers would not churn, and they actually didn’t.
# 
# False Positives (FP = 94): The model predicted 94 customers would churn, but they actually stayed. These are false alarms.
# 
# False Negatives (FN = 257): 257 churners were missed — the model predicted they would stay.
# 
# True Positives (TP = 252): The model correctly identified 252 customers who actually churned.

# #### XGBoost Model

# In[51]:


# Instantiating the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# Creating a dictionary of hyperparameters to tune
cv_params = {'max_depth': [6, 12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300]
             }

# Defining a dictionary of scoring metrics to capture
scoring = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'precision': 'precision'
}


# Instantiating the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='recall')


# In[53]:


start = time.time()
xgb_cv.fit(X_train, y_train)
end = time.time()
print(f"Training time: {end - start:.2f} seconds")


# In[54]:


# Examining best score
xgb_cv.best_score_


# In[55]:


# Examining best parameters
xgb_cv.best_params_


# In[56]:


def make_results(model_name, model, refit_metric):
    '''
    Extract cross-validated results from a GridSearchCV object.

    Parameters:
        model_name (str): name of the model to appear in the results
        model (GridSearchCV): a fitted GridSearchCV object
        refit_metric (str): the metric that was used to refit the model

    Returns:
        pd.DataFrame: with precision, recall, f1, accuracy, and refit_metric scores
    '''
    results_df = pd.DataFrame({
        'model': [model_name],
        'precision': [model.cv_results_[f'mean_test_precision'][model.best_index_]],
        'recall':    [model.cv_results_[f'mean_test_recall'][model.best_index_]],
        'f1':        [model.cv_results_[f'mean_test_f1'][model.best_index_]],
        'accuracy':  [model.cv_results_[f'mean_test_accuracy'][model.best_index_]],
        refit_metric: [model.best_score_]
    })

    return results_df


# In[57]:


# Generate the results DataFrame for XGBoost
xgb_cv_results = make_results('XGB cv', xgb_cv, 'recall')

# If this is your first model result, initialize 'results' with it
results = xgb_cv_results

# Display results
results


# The tuned XGBoost model achieved a recall of 49.3%, accuracy of 85%, precision of 68%, and F1-score of 57% using recall as the refit metric.
# 
# Best hyperparameters: learning_rate=0.1, max_depth=12, min_child_weight=3, and n_estimators=300.
# 
# XGBoost outperformed logistic regression and is comparable to Random Forest, offering better recall and class balance for predicting churn.

# In[58]:


# Extract feature importances from the best estimator
xgb_best = xgb_cv.best_estimator_

# Create a DataFrame with feature importances
xgb_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_best.feature_importances_
}).sort_values(by='importance', ascending=False)


# #### Feature Importances

# In[59]:


plt.figure(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=xgb_importances, color='skyblue')
plt.title("Feature Importance from XGBoost Model", fontsize=12)
plt.xlabel("Importance Score", fontsize=11)
plt.ylabel("Feature", fontsize=11)
plt.tight_layout()
plt.show()


# NumOfProducts, IsActiveMemeber, Geography_Germany, and Age emerge as the most influential features, with Fare exhibiting high variability.

# #### Summary of the Model

# The XGBoost classifier was trained and tuned using GridSearchCV with a focus on optimizing recall. After testing multiple hyperparameter combinations, the best model achieved:
# 
# Accuracy: 84.97%
# 
# Recall: 49.28%
# 
# Precision: 68.14%
# 
# F1 Score: 57.17%
# 
# Best parameters: learning_rate=0.1, max_depth=12, min_child_weight=3, n_estimators=300

# #### Conclusion

# The XGBoost model achieved solid predictive performance, with an accuracy of ~85% and recall of ~49% in identifying customers likely to churn. Feature importance analysis revealed that NumOfProducts, IsActiveMember, Geography_Germany, and Age are the most influential predictors of churn. This suggests that product engagement, customer activity level, demographics, and regional behaviors play significant roles in churn behavior.

# In[ ]:




