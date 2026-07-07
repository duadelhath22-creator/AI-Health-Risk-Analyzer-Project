import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(data.feature_names)
import pandas as pd

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)


df["target"] = data.target
X = df.drop("target", axis=1)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier()

model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)
param_grid = {
    "n_estimators":[100,200],
    "max_depth":[5,10]
}
import joblib

joblib.dump(model, "models/model.pkl")