import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# Load dataset
df = pd.read_csv("data/Employee.csv")

# Features and target
X = df.drop("Age", axis=1)
y = df["Age"]

# Numeric and categorical columns
numeric_features = [
    "JoiningYear",
    "PaymentTier",
    "ExperienceInCurrentDomain"
]

categorical_features = [
    "Education",
    "City",
    "Gender",
    "EverBenched"
]

# Preprocessing
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Pipeline
linear_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
linear_pipeline.fit(X_train_reg, y_train_reg)

# Predict
y_pred = linear_pipeline.predict(X_test_reg)

# Metrics
mae = mean_absolute_error(y_test_reg, y_pred)
mse = mean_squared_error(y_test_reg, y_pred)
r2 = r2_score(y_test_reg, y_pred)

print("\n--- Linear Regression ---")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# Ridge Regression

ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

ridge_pipeline.fit(X_train_reg, y_train_reg)

ridge_pred = ridge_pipeline.predict(X_test_reg)

ridge_mae = mean_absolute_error(y_test_reg, ridge_pred)
ridge_mse = mean_squared_error(y_test_reg, ridge_pred)
ridge_r2 = r2_score(y_test_reg, ridge_pred)

print("\n--- Ridge Regression ---")
print("MAE:", ridge_mae)
print("MSE:", ridge_mse)
print("R2 Score:", ridge_r2)


# Lasso Regression

lasso_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Lasso(alpha=0.1, max_iter=10000))
])

lasso_pipeline.fit(X_train_reg, y_train_reg)

lasso_pred = lasso_pipeline.predict(X_test_reg)

lasso_mae = mean_absolute_error(y_test_reg, lasso_pred)
lasso_mse = mean_squared_error(y_test_reg, lasso_pred)
lasso_r2 = r2_score(y_test_reg, lasso_pred)

print("\n--- Lasso Regression ---")
print("MAE:", lasso_mae)
print("MSE:", lasso_mse)
print("R2 Score:", lasso_r2)
