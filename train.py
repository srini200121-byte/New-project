import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from preprocessing import preprocessor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#load dataset

df = pd.read_csv("data/Employee.csv")
df["YearsSinceJoining"] = 2026 - df["JoiningYear"]

df["PaymentExperience"] = (
    df["PaymentTier"] * df["ExperienceInCurrentDomain"]
)

df["LogAge"] = np.log1p(df["Age"])

print("\n--- Feature Engineering ---")
print(df[[
    "JoiningYear",
    "YearsSinceJoining",
    "PaymentTier",
    "ExperienceInCurrentDomain",
    "PaymentExperience",
    "Age",
    "LogAge"
]].head())


#display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")

print(df["LeaveOrNot"].value_counts())

#features and target

X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]

#First split the 80% temporary data and 20% test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Second split: 60% train, 20% validation
# 25% of the remaining 80% = 20% of total data

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Display sizes
print("\nTrain shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

#check target distribution

print("\nTrain target distrubution:")
print(y_train.value_counts(normalize=True))

print("\nValidation target distribution:")
print(y_val.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))


#Temporarity encode categorical columns

X_train_encoded = pd.get_dummies(X_train)
X_val_encoded = pd.get_dummies(X_val)

# Make sure both have the same columns
X_train_encoded, X_val_encoded = X_train_encoded.align(
    X_val_encoded,
    join="left",
    axis=1,
    fill_value=0
)
depths = [1, 3, 5, 10, None]

train_accuracies = []
val_accuracies = []

print("\n--- Decision Tree Overfitting Experiment ---")

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train_encoded, y_train)


    train_pred = model.predict(X_train_encoded)
    val_pred = model.predict(X_val_encoded)

    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(
        f"Depth {depth}: "
        f"train {train_accuracy:.2f} "
        f"val {val_accuracy:.2f}"
    )
depth_labels = ["1","3","5","10","None"]
plt.figure(figsize=(8,5))

plt.plot(
    depth_labels,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    depth_labels,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)


plt.savefig("plots/overfitting.png")
plt.show()



#Combine Train + Validation data
X_cv = pd.concat([X_train, X_val])
y_cv = pd.concat([y_train, y_val])

# Convert categorical columns to numbers
X_cv_encoded = pd.get_dummies(X_cv)

# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(
    logistic_model,
    X_cv_encoded,
    y_cv,
    cv=5,
    scoring="accuracy"
)

print("\n--- 5-Fold Cross-Validation ---")
print("CV scores:", cv_scores)
print("CV mean:", cv_scores.mean())
print("CV standard deviation:", cv_scores.std())


X_train_encoded = pd.get_dummies(X_train)
X_val_encoded = pd.get_dummies(X_val)


# Make sure both have the same columns
X_train_encoded, X_val_encoded = X_train_encoded.align(
    X_val_encoded,
    join="left",
    axis=1,
    fill_value=0
)

# WRONG: Scaling before split
# -----------------------------
scaler_wrong = StandardScaler()
X_all = pd.concat([X_train_encoded, X_val_encoded])

X_all_scaled = scaler_wrong.fit_transform(X_all)

X_train_wrong = X_all_scaled[:len(X_train_encoded)]
X_val_wrong = X_all_scaled[len(X_train_encoded):]

model_wrong = LogisticRegression(max_iter=1000)
model_wrong.fit(X_train_wrong, y_train)

wrong_pred = model_wrong.predict(X_val_wrong)

wrong_accuracy = accuracy_score(y_val, wrong_pred)

print("\n--- Data Leakage Demonstration ---")
print("Accuracy with leakage:", wrong_accuracy)

# CORRECT: Scaling only train
# -----------------------------
scaler_correct = StandardScaler()

X_train_scaled = scaler_correct.fit_transform(X_train_encoded)
X_val_scaled = scaler_correct.transform(X_val_encoded)


model_correct = LogisticRegression(max_iter=1000)
model_correct.fit(X_train_scaled, y_train)
correct_pred = model_correct.predict(X_val_scaled)

correct_accuracy = accuracy_score(y_val, correct_pred)

print("Accuracy without leakage:", correct_accuracy)


#separate features and target
X = df.drop("LeaveOrNot", axis=1)
y= df["LeaveOrNot"]

#Numeric columns
numeric_features = [
    "JoiningYear",
    "PaymentTier",
    "Age",
    "ExperienceInCurrentDomain"
]

categorical_features = [
    "Education",
    "City",
    "Gender",
    "EverBenched"
]

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
# Combine both
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\n--- Preprocessing Pipeline Created Successfully ---")


logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

#Train 
logistic_pipeline.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_pipeline,
    "models/logistic_pipeline.pkl"
)

print("Logistic Regression Pipeline saved successfully.")
# Validation prediction
val_predictions = logistic_pipeline.predict(X_val)

# Accuracy
pipeline_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print("\n--- Logistic Regression Pipeline ---")
print("Validation Accuracy:", pipeline_accuracy)
# TASK 2 - Step 4: KNN Imputer
# ==========================================

knn_imputer = KNNImputer(n_neighbors=5)

#Apply KNN imputer to numeric training data
X_train_numeric = X_train[numeric_features].copy()
X_val_numeric = X_val [numeric_features].copy()

X_train_knn = knn_imputer.fit_transform(X_train_numeric)
X_val_knn = knn_imputer.transform(X_val_numeric)

print("\n--- KNN Imputer ---")
print("KNN imputation completed successfully.")
print("Training shape:", X_train_knn.shape)
print("Validation shape:", X_val_knn.shape)
#knn imputer code

ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_categorical = X_train[categorical_features]
X_val_categorical = X_val[categorical_features]

X_train_ordinal = ordinal_encoder.fit_transform(
    X_train_categorical
)

X_val_ordinal = ordinal_encoder.transform(
    X_val_categorical
)

print("\n---ordinal Encoding---")
print("Training shape: ", X_train_ordinal.shape)
print("Validation shapeL:", X_val_ordinal.shape )
#Feature Engineering

df["YearsSinceJoining"] = 2026 -df["JoiningYear"]

df["PaymentExperience"] =(
    df["PaymentTier"] * df["ExperienceInCurrentDomain"]

)

df["LogAge"] = np.log1p(df["Age"])

print("\n----Feature Engineering-----")
print(df[[
    "JoiningYear",
    "YearsSinceJoining",
    "PaymentTier",
    "ExperienceInCurrentDomain",
    "PaymentExperience",
    "Age",
    "LogAge"
]].head())


decision_tree_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

decision_tree_pipeline.fit(X_train, y_train)
dt_predictions = decision_tree_pipeline.predict(X_val)

dt_accuracy = accuracy_score(
    y_val,
    dt_predictions
)

print("\n---Decision Tree Pipeline---")
print("Validation Accuracy:", dt_accuracy)
random_forest_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_pipeline.fit(X_train, y_train)

rf_predictions = random_forest_pipeline.predict(X_val)

rf_accuracy = accuracy_score(
    y_val,
    rf_predictions
)

print("\n--- Random Forest Pipeline ---")
print("Validation Accuracy:", rf_accuracy)

rf_model = random_forest_pipeline.named_steps["model"]
print("OOB Score:", rf_model.oob_score_)

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators =200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_pipeline.fit(X_train, y_train)
xgb_predictions = xgb_pipeline.predict(X_val)
xgb_accuracy = accuracy_score(
    y_val,
    xgb_predictions
)
print("\n---XGBoost Pipeline---")

print("Validation Accuracy:", xgb_accuracy)

# Probability predictions for 5 validation employees


rf_probabilities_5 = random_forest_pipeline.predict_proba(X_val.head(5))

print("\n--- Random Forest Probability Predictions ---")

for i, probabilities in enumerate(rf_probabilities_5, start=1):
    print(
        f"Employee {i}: "
        f"Stay={probabilities[0]:.3f}, "
        f"Leave={probabilities[1]:.3f}"
    )
rf_probabilities = random_forest_pipeline.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, rf_probabilities)
# Feature Importance


rf_model = random_forest_pipeline.named_steps["model"]
rf_preprocessor = random_forest_pipeline.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()

importance_values = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Random Forest Feature Importance ---")
print(feature_importance.head(15))

#classification Metrics

accuracy = accuracy_score(y_val,rf_predictions)

precision = precision_score(y_val,rf_predictions)

recall = recall_score(y_val, rf_predictions)
f1 = f1_score(y_val, rf_predictions)

cm = confusion_matrix(y_val, rf_predictions)

print("\n---Random Forest Classification Metrics----")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(cm)

target_recall = 0.75
best_threshold = None
best_precision = 0

for threshold in thresholds:
    threshold_predictions = (rf_probabilities >= threshold).astype(int)

    recall = recall_score(y_val, threshold_predictions, zero_division=0)
    
    precision = precision_score(
        y_val,
        threshold_predictions,
        zero_division=0
    )

    if recall >= target_recall and precision > best_precision:
        best_precision = precision
        best_threshold = threshold
print("\n--- Threshold Analysis ---")

if best_threshold is not None:

    final_predictions = (
        rf_probabilities >= best_threshold
    ).astype(int)

    final_recall = recall_score(
        y_val,
        final_predictions,
        zero_division=0
    )

    final_precision = precision_score(
        y_val,
        final_predictions,
        zero_division=0
    )

    print("Best Threshold:", best_threshold)
    print("Recall:", final_recall)
    print("Precision:", final_precision)

else:
    print("No threshold achieved 75% recall.")


# ==========================================
# Random Forest GridSearchCV
# ==========================================

from sklearn.model_selection import GridSearchCV

rf_grid_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        oob_score=True,
        n_jobs=-1
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_grid_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n--- Random Forest Grid Search ---")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

grid_rf_predictions = grid_search.predict(X_val)

grid_rf_accuracy = accuracy_score(
    y_val,
    grid_rf_predictions
)

print("Validation Accuracy:", grid_rf_accuracy)

# ==========================================
# XGBoost RandomizedSearchCV
# ==========================================

from sklearn.model_selection import RandomizedSearchCV

xgb_random_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_param_distributions = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    xgb_random_pipeline,
    xgb_param_distributions,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\n--- XGBoost Randomized Search ---")
print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

random_xgb_predictions = random_search.predict(X_val)

random_xgb_accuracy = accuracy_score(
    y_val,
    random_xgb_predictions
)

print("Validation Accuracy:", random_xgb_accuracy)
# Mean prediction baseline
mean_prediction = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    mean_prediction
)

baseline_mse = mean_squared_error(
    y_test,
    mean_prediction
)

baseline_r2 = r2_score(
    y_test,
    mean_prediction
)

print("\n--- Mean Baseline ---")
print("MAE:", baseline_mae)
print("MSE:", baseline_mse)
print("R2:", baseline_r2)

# ==========================================
# Final Model Evaluation on Test Set
# ==========================================

final_model = grid_search.best_estimator_

final_test_predictions = final_model.predict(X_test)

final_test_accuracy = accuracy_score(
    y_test,
    final_test_predictions
)

final_test_precision = precision_score(
    y_test,
    final_test_predictions,
    zero_division=0
)

final_test_recall = recall_score(
    y_test,
    final_test_predictions,
    zero_division=0
)

final_test_f1 = f1_score(
    y_test,
    final_test_predictions,
    zero_division=0
)

final_test_cm = confusion_matrix(
    y_test,
    final_test_predictions
)

print("\n--- Final Model Test Evaluation ---")
print("Test Accuracy:", final_test_accuracy)
print("Test Precision:", final_test_precision)
print("Test Recall:", final_test_recall)
print("Test F1 Score:", final_test_f1)

print("\nTest Confusion Matrix:")
print(final_test_cm)

os.makedirs("models", exist_ok=True)

joblib.dump(
    final_model,
    "models/final_employee_leave_model.pkl"
)

print("\nFinal model saved successfully.")
print("Location: models/final_employee_leave_model.pkl")
