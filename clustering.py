import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("data/Employee.csv")

# Select numerical features
X = df[
    [
        "JoiningYear",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain"
    ]
]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Store results
k_values = range(2, 9)
inertias = []
silhouette_scores = []

# K-Means
for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

# Print results
print("\n--- K-Means Results ---")

for k, inertia, silhouette in zip(
    k_values,
    inertias,
    silhouette_scores
):
    print(
        f"K={k} | "
        f"Inertia={inertia:.2f} | "
        f"Silhouette={silhouette:.3f}"
    )

# Create plots folder
import os
os.makedirs("plots", exist_ok=True)

# Elbow Plot
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.xticks(list(k_values))
plt.grid(True)

plt.savefig("plots/kmeans_elbow.png")
plt.show()

print("\nElbow plot saved successfully.")


#Silhouette Score plot

plt.figure(figsize=(8,5))
plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)
best_k = list(k_values)[
    silhouette_scores.index(max(silhouette_scores))
]

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("k-Means Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.savefig("plots/kmeans_silhouette.png")
plt.show()

print("Silhouette Plot Saved successfully.")

best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]

print("Best K based on Silhouette Score:", best_k)