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
