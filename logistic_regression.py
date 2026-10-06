import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, confusion_matrix

# 1. Load the dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Convert gender to numbers: Female = 0, Male = 1
data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1})

# Select the inputs and output
X = data[["Age", "EstimatedSalary", "Gender"]].to_numpy(dtype=float)
y = data["Purchased"]

# 2. Use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale age and salary
# Learn the scaling from the training data only
scaler = StandardScaler()
X_train[:, :2] = scaler.fit_transform(X_train[:, :2])
X_test[:, :2] = scaler.transform(X_test[:, :2])

# 4. Train the logistic regression model
model = LogisticRegression(
    solver="lbfgs", C=1.0, max_iter=1000, random_state=42
)
model.fit(X_train, y_train)

# 5. Predict the test data
predictions = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")

print("\nConfusion Matrix:")
print("Rows = actual, columns = predicted")
print("Class order: 0 = no purchase, 1 = purchase")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 7. Create a new observation
# Age = 40, salary = 90000, gender = Female (0)
new_person = pd.DataFrame(
    [[40, 90000, 0]],
    columns=["Age", "EstimatedSalary", "Gender"]
).to_numpy(dtype=float)

# Apply the same scaling
new_person[:, :2] = scaler.transform(new_person[:, :2])

# 8. Predict the new observation
prediction = model.predict(new_person)[0]
probability = model.predict_proba(new_person)[0, 1]

print("\nNew person: Female, age 40, salary 90000")
print("Predicted Purchased:", prediction)
print(f"Purchase probability: {probability:.2%}")

# Results:
# Accuracy: 81.25%
# Precision: 81.82%
# Recall: 62.07%
#
# Confusion Matrix:
# [[47  4]
#  [11 18]]
#
# New prediction: 1 (purchase)
# Purchase probability: 50.89%
#
# Conclusion:
# The model correctly predicted 65 out of 80 test observations.
# It missed 11 actual purchasers.
# The new person was predicted to purchase, but the probability
# was only slightly above 50%, so this prediction is uncertain.
