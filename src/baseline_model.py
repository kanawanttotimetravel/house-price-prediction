import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# training model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluating
y_valid_pred = model.predict(X_valid)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
r2_test = r2_score(y_valid, y_valid_pred)

print(f'Validation MSE: {mse_valid}')
print(f'Validation r2: {r2_test}')

# evaluate on test data
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test MSE: {mse_test}')
print(f'Test r2: {r2_test}')

# Validation MSE: 58543793390.35076
# Validation r2: 0.6813692149597749
# Test MSE: 39278274898.18614
# Test r2: 0.6681538561134266

# optimize
# Test MSE: 39271537635.337975
# Test r2: 0.6682107765026828