import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")

print("Dataset loaded successfully!")

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Scale Time and Amount
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# -----------------------------
# 1. Logistic Regression
# -----------------------------
logistic = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)

# -----------------------------
# 2. Decision Tree
# -----------------------------
tree = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=10
)

tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# -----------------------------
# 3. Random Forest
# -----------------------------
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
models = {
    "Logistic Regression": logistic_pred,
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred
}

print("\n========== MODEL COMPARISON ==========")

for name, predictions in models.items():

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n" + name)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    import joblib

# Save the trained Random Forest model
joblib.dump(forest, "model/fraud_detection_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler saved successfully!")

print("\nRandom Forest model saved successfully!")