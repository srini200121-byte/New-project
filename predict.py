import joblib
import pandas as pd
import numpy as np

#Load Saved  pipeline

model = joblib.load("models/logistic_pipeline.pkl")

#3 new employees

new_employees = pd.DataFrame([
    {
        "Education": "Bachelors",
        "JoiningYear": 2018,
        "City":"Bangalore",
        "PaymentTier":2,
         "Age": 25,
        "Gender": "Male",
        "EverBenched": "No",
        "ExperienceInCurrentDomain": 2
    },
    {
        "Education": "Masters",
        "JoiningYear":2014,
        "city":"Pune",
         "PaymentTier": 1,
        "Age": 35,
        "Gender": "Female",
        "EverBenched": "Yes",
        "ExperienceInCurrentDomain": 8
    },
    {
        "Education": "Bachelors",
        "JoiningYear": 2020,
        "City": "New Delhi",
        "PaymentTier": 3,
        "Age": 23,
        "Gender": "Male",
        "EverBenched": "No",
        "ExperienceInCurrentDomain": 1
    }
])

#Fearture Engineering
new_employees["YearsSinceJoining"]=(
    2026 - new_employees["JoiningYear"]
)
new_employees["PaymentExperience"]=(
    new_employees["PaymentTier"]
    * new_employees["ExperienceInCurrentDomain"]
)

new_employees["LogAge"] = np.log1p(
    new_employees["Age"]
)

# Prediction
predictions = model.predict(new_employees)

print("\n--- Predictions ---")

for i, prediction in enumerate(predictions, start=1):
    result = "LEAVE" if prediction == 1 else "STAY"
    print(f"Employee {i}: {result}")

# Load final model
model = joblib.load(
    "models/final_employee_leave_model.pkl"
)
print("\n--- Employee Leave Prediction ---")

education = input("Education (Bachelors/Masters/PHD): ")
joining_year = int(input("Joining Year: "))
city = input("City (Bangalore/Pune/New Delhi): ")
payment_tier = int(input("Payment Tier (1/2/3): "))
age = int(input("Age: "))
gender = input("Gender (Male/Female): ")
ever_benched = input("Ever Benched (Yes/No): ")
experience = int(input("Experience In Current Domain: "))


# New employee
new_employee = pd.DataFrame([{
    "Education": "Bachelors",
    "JoiningYear": 2020,
    "City": "Bangalore",
    "PaymentTier": 2,
    "Age": 28,
    "Gender": "Male",
    "EverBenched": "No",
    "ExperienceInCurrentDomain": 3
}])

# Feature engineering
new_employee["YearsSinceJoining"] = (
    2026 - new_employee["JoiningYear"]
)

new_employee["PaymentExperience"] = (
    new_employee["PaymentTier"]
    * new_employee["ExperienceInCurrentDomain"]
)

new_employee["LogAge"] = np.log1p(
    new_employee["Age"]
)

# Prediction
prediction = model.predict(new_employee)[0]

# Probability
probability = model.predict_proba(new_employee)[0]

print("\n--- Employee Leave Prediction ---")

if prediction == 1:
    print("Prediction: LEAVE")
else:
    print("Prediction: STAY")

print(f"Stay Probability: {probability[0]:.3f}")
print(f"Leave Probability: {probability[1]:.3f}")
