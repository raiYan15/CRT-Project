# CRT-Project
# 1. Project Content
Develop a machine learning model to predict possible diseases based on patient health-related data.

# 2. Project Code

~~~python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_csv("healthcare_dataset.csv")
df.drop(columns=['Name'], inplace=True)
label_encoders = {col: LabelEncoder().fit(df[col]) for col in df.select_dtypes(include=['object']).columns}
for col, le in label_encoders.items():
    df[col] = le.transform(df[col])

X = df.drop("Disease", axis=1)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

with open('disease_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
~~~

# 3. Key Technologies
- Python
- Pandas, NumPy (Data manipulation)
- Scikit-learn (Logistic Regression, Label Encoding, Evaluation)
- Pickle (Model saving)

# 4. Description
This project aims to simplify early diagnosis in healthcare. It uses logistic regression, a classification algorithm, to predict diseases based on features like symptoms, age, and gender. The data is preprocessed to remove non-numerical or irrelevant fields, and encoded for training the model. The model is then evaluated and saved for deployment.

# 5. Output
- Accuracy score of the model.
- Classification report showing precision, recall, and F1-score.
- A saved .pkl file that can be used in healthcare systems to make predictions on new patient data.

# 6. Further Research
- Expand dataset to include more symptoms and rare diseases.
- Try ensemble models like Random Forest or XGBoost.
- Build a web interface using Flask or Django.
- Integrate real-time patient data using IoT or EHRs.
