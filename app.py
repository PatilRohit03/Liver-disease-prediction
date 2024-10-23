# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
url = 'indian_liver_patient.csv'  # Upload your file to Google Colab and provide its path here
data = pd.read_csv(url)

# Data Preprocessing
# Check for missing values

# Drop rows with missing values (or you can fill them with mean/median values if preferred)
data.dropna(inplace=True)

# Encode categorical variables (Gender)
labelencoder = LabelEncoder()
data['Gender'] = labelencoder.fit_transform(data['Gender'])  # Convert Female/Male to 0/1

# Separate features and target variable
X = data.drop(columns=['Dataset'])  # Features
y = data['Dataset']  # Target

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Model Prediction
y_pred = rf_classifier.predict(X_test)

# Model Evaluation
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
