

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Handle missing values by filling them with the mean of each column
data.fillna(data.mean(), inplace=True)

# Features and target variable
X = data.drop(columns=['case', 'bwt'])
y = data['bwt']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Apply Recursive Feature Elimination (RFE) with different regressors
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(kernel='linear')
}

rfe_results = {}

for model_name, model in models.items():
    rfe = RFE(model, n_features_to_select=1)
    rfe.fit(X_train, y_train)
    
    # Rank features
    rfe_results[model_name] = {
        'ranking': rfe.ranking_,
        'support': rfe.support_
    }
    
    # Predictions
    y_pred = rfe.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rfe_results[model_name]['mse'] = mse

# Display RFE results
rfe_results

'''Linear Regression:
Feature Rankings: [1, 4, 6, 3, 5, 2]
Support (Selected Feature): [True, False, False, False, False, False]
Mean Squared Error (MSE): 312.28
Random Forest:
Feature Rankings: [1, 6, 3, 4, 2, 5]
Support (Selected Feature): [True, False, False, False, False, False]
Mean Squared Error (MSE): 321.80
Support Vector Regressor (SVR):
Feature Rankings: [1, 5, 6, 3, 4, 2]
Support (Selected Feature): [True, False, False, False, False, False]
Mean Squared Error (MSE): 309.03
In all three models, the feature gestation (first feature) was identified as the most significant feature. The rankings indicate the order of feature importance, with 1 being the most important. The MSE values show the performance of each model with the selected features.

To summarize, gestation appears to be the most important feature for predicting bwt (birth weight) across all three regression algorithms'''


'''Calculate R² values for each model.
Use RFECV to select the optimal number of features.'''
from sklearn.feature_selection import RFECV

# Initialize a dictionary to store the results
rfecv_results = {}

# Perform RFECV with each model
for model_name, model in models.items():
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='r2')
    rfecv.fit(X_train, y_train)
    
    # Rank features
    rfecv_results[model_name] = {
        'ranking': rfecv.ranking_,
        'support': rfecv.support_,
        'n_features': rfecv.n_features_
    }
    
    # Predictions
    y_pred = rfecv.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = rfecv.score(X_test, y_test)
    rfecv_results[model_name]['mse'] = mse
    rfecv_results[model_name]['r2'] = r2

# Display RFECV results
rfecv_results
'''Based on the RFECV results, we have the following information for each model:

Linear Regression:
Feature Rankings: [1, 1, 2, 1, 1, 1]
Selected Features: [True, True, False, True, True, True]
Number of Features Selected: 5
Mean Squared Error (MSE): 278.81
R² Value: 0.293
Random Forest:
Feature Rankings: [1, 1, 1, 1, 1, 1]
Selected Features: [True, True, True, True, True, True]
Number of Features Selected: 6
Mean Squared Error (MSE): 307.23
R² Value: 0.221
Support Vector Regressor (SVR):
Feature Rankings: [1, 1, 2, 1, 1, 1]
Selected Features: [True, True, False, True, True, True]
Number of Features Selected: 5
Mean Squared Error (MSE): 276.13
R² Value: 0.300
Summary:
Linear Regression and Support Vector Regressor selected 5 features, excluding the third feature (parity).
Random Forest selected all 6 features.
The SVR model achieved the highest R² value (0.300), indicating the best fit among the models, followed closely by Linear Regression (0.293).
Random Forest had the lowest R² value (0.221) and the highest MSE (307.23).
Selected Features for Linear Regression and SVR:
gestation
age
height
weight
smoke
These results suggest that the parity feature might not be as significant in predicting bwt (birth weight) for Linear Regression and SVR models, as it was excluded during the feature selection process.
''''
