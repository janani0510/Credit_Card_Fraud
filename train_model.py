
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

data = pd.read_csv("data//compressed_data.csv.gz")
x = data.drop("Class",axis=1)
y = data["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000,class_weight="balanced", solver='lbfgs')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred, digits=4))
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")

joblib.dump(model,"fraud_model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("\n Model And Scaler Saved Successfully!")
