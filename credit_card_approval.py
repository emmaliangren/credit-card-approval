# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None) 

# Preprocessing the data
cc_apps_prep=cc_apps.replace("?", np.nan)

# Imputing based on data type
cc_apps_imputed=cc_apps_prep.copy()
for col in cc_apps_imputed.columns:
    if cc_apps_imputed[col].dtypes == "object":
        cc_apps_imputed[col].fillna(cc_apps_imputed[col].value_counts().index[0])
    else:
        cc_apps_imputed[col].fillna(cc_apps_imputed[col].mean())

# One-hot encoding
cc_apps_encoded = pd.get_dummies(cc_apps_imputed, drop_first = True)

# Define target variable
X= cc_apps_encoded.iloc[:, :-1].values
y= cc_apps_encoded.iloc[:, [-1]].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=77)

# Scaling data
scaler = StandardScaler()
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)

# Fit and predict
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)
pred_y_train = logreg.predict(rescaledX_train)

print(confusion_matrix(y_train, pred_y_train))

# Grid search cross validation
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

param_grid = dict(tol=tol, max_iter=max_iter)
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
grid_model_result = grid_model.fit(rescaledX_train, y_train)

# Results/Grid search best model
best_train_score, best_train_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_train_score, best_train_params))

best_model = grid_model_result.best_estimator_
best_score =  best_model.score(rescaledX_test, y_test)
print("Accuracy of logistic regression classifier: ", best_score)