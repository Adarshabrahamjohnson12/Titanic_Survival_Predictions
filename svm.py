import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("train.csv")

# Clean & simplify: use only 2 features (Age and Fare)
df = df[["Survived", "Sex", "Age", "Fare"]]
df = df.dropna()
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

X = df[["Age", "Fare"]]
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train SVM (linear for easy plot)
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# Plotting
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X_train["Age"], X_train["Fare"], c=y_train, cmap='coolwarm', edgecolors='k')

# Create meshgrid
import numpy as np
xx, yy = np.meshgrid(
    np.linspace(X["Age"].min(), X["Age"].max(), 500),
    np.linspace(X["Fare"].min(), X["Fare"].max(), 500)
)

# Predict over grid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("SVM Decision Boundary (Age vs Fare)")
plt.colorbar(label="Survival (0 = No, 1 = Yes)")
plt.grid(True)
plt.show()
