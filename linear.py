from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
data3 = fetch_california_housing()
X3, y3 = data3.data, data3.target

# Train
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
model3 = LinearRegression()
model3.fit(X3_train, y3_train)

print("R2 Score:", model3.score(X3_test, y3_test))

# Save
joblib.dump(model3, "linear_model.pkl")