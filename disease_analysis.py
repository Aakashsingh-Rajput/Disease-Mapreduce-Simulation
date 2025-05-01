import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'disease_data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing

# Encoding categorical columns
label_encoder = LabelEncoder()

# Encode categorical columns like 'Fever', 'Cough', etc. as needed
data['Fever'] = label_encoder.fit_transform(data['Fever'])
data['Cough'] = label_encoder.fit_transform(data['Cough'])
data['Fatigue'] = label_encoder.fit_transform(data['Fatigue'])
data['Difficulty Breathing'] = label_encoder.fit_transform(data['Difficulty Breathing'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Blood Pressure'] = label_encoder.fit_transform(data['Blood Pressure'])
data['Cholesterol Level'] = label_encoder.fit_transform(data['Cholesterol Level'])
data['Outcome Variable'] = label_encoder.fit_transform(data['Outcome Variable'])  # Target variable

# Step 3: Split data into features and target
X = data.drop(columns=['Disease', 'Outcome Variable'])  # Dropping 'Disease' and 'Outcome Variable'
y = data['Outcome Variable']  # Target variable

# Step 4: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions and Evaluation

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Additional Visualizations

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Step 8: Disease Frequency Visualization
disease_counts = data['Disease'].value_counts()

# Select the top 5 and least 5 diseases
top_5_diseases = disease_counts.head(5)
least_5_diseases = disease_counts.tail(5)

# Combine top and least diseases for plotting
combined_diseases = pd.concat([top_5_diseases, least_5_diseases])

# Create a bar plot for disease frequency count of top 5 and least 5 diseases
plt.figure(figsize=(10, 6))
sns.barplot(x=combined_diseases.index, y=combined_diseases.values, palette='Set2')
plt.title('Frequency Count of Top 5 and Least 5 Diseases')
plt.xlabel('Disease')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()
