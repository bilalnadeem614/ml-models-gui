from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = load_iris()
X, y = data.data, data.target

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, y_train)

print("Accuracy:", model1.score(X_test, y_test))

# Save
joblib.dump(model1, "logistic_model.pkl")