from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
data2 = load_breast_cancer()
X2, y2 = data2.data, data2.target

# Train
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = DecisionTreeClassifier(max_depth=4, random_state=42)
model2.fit(X2_train, y2_train)

print("Accuracy:", model2.score(X2_test, y2_test))

# Save
joblib.dump(model2, "decisiontree_model.pkl")