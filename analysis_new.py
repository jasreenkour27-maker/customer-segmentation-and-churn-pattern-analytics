import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("European_Bank.csv")
print(df.columns)
print(df.head())
# Correlation heatmap
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Age
plt.figure(figsize=(8,5))
sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Age by Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Age")
plt.show()

# Balance
plt.figure(figsize=(8,5))
sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance by Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Balance")
plt.show()
plt.ylabel("Credit Score")
plt.show()

# Age Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Age by Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Age")
plt.show()

# Balance Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance by Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Balance")
plt.show()
from sklearn.model_selection import train_test_split
X = df.drop("Exited", axis=1)
y = df["Exited"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()