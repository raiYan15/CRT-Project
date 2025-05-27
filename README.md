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


# Project 2: Dog and Cat Image Classification Using CNN

# 1. Project Content
This project focuses on detecting whether an image contains a dog or a cat using a machine learning model. It involves loading datasets, preprocessing images, training a model, and evaluating its accuracy.

# 2. Project Code

~~~python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set image dimensions and paths
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Building the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Plotting accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
~~~

# 3. Key Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Deep Learning (CNN)

# 4. Description
The goal of the project is to develop a classification model that can accurately distinguish between dog and cat images. The dataset is split into training and validation sets, with images being resized and normalized. A CNN model is constructed with multiple convolutional, pooling, and dense layers. The model is trained using binary crossentropy as the loss function and accuracy as the metric.

# 5. Output
- Accuracy and loss plots
- Example predictions on validation/test images showing whether the image is classified as a dog or a cat
- Final model accuracy

# 6. Further Research
- Improve accuracy using more advanced architectures like ResNet, Inception, or EfficientNet
- Hyperparameter tuning for optimal performance
- Implement transfer learning with pre-trained models
- Expand to multi-class classification including more animal types
- Deploy the model as a web app using Flask or Streamlit
