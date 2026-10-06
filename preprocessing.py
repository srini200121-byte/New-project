from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Numeric columns
numeric_features = [
    "JoiningYear",
    "PaymentTier",
    "Age",
    "ExperienceInCurrentDomain",
    "YearsSinceJoining",
    "PaymentExperience",
    "LogAge"
]


# Categorical columns
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


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)