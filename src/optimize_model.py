# base on baseline_model

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Using cross-validation + normalize dataset + ridge

# loading data
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
valid = pd.read_csv("Data/valid.csv")

drop_col = ['id', 'date']
train = train.drop(columns=drop_col)
valid = valid.drop(columns=drop_col)
test = test.drop(columns=drop_col)

X_train = train.drop(columns=['price'])
y_train = train['price']

X_valid = valid.drop(columns=['price'])
y_valid = valid['price']

X_test = test.drop(columns=['price'])
y_test = test['price']

# normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge()

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
    # 'alpha':  [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

model = grid_search.best_estimator_

# evaluating
y_valid_pred = model.predict(X_valid_scaled)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
r2_test = r2_score(y_valid, y_valid_pred)

print(f'Validation MSE: {mse_valid}')
print(f'Validation r2: {r2_test}')

y_test_pred = model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test MSE: {mse_test}')
print(f'Test r2: {r2_test}')