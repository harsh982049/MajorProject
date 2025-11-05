import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ----------------------------
# Load the session feature data
# ----------------------------
df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\session_features.csv")

# print(df['stress_label'].unique())
df['stress_binary'] = df['stress_label'].apply(lambda x: 1 if 'Stressed' in x else 0)
# print(df[['stress_binary', 'stress_label']])

# Encode the categorical target label
# label_encoder = LabelEncoder()
# df['stress_label_encoded'] = label_encoder.fit_transform(df['stress_label'])

# Define feature columns
feature_columns = [
    'avg_keypress_duration', 'keypress_count', 'backspace_count', 'error_rate',
    'avg_mouse_speed', 'mouse_move_count', 'mouse_click_count',
    'hour', 'day_of_week', 'daylight_morning', 'daylight_evening', 'session_active'
]

# Prepare feature matrix X and label vector y
X = df[feature_columns]
y = df['stress_binary'] # This is For binary classification
# y = df['stress_label_encoded'] # This is For normal (the default classes which are present) classification

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train) # Without resampling
# clf.fit(X_train_resampled, y_train_resampled) # With resampling

# # Save the model and label encoder using pickle
# with open("rf_stress_model.pkl", "wb") as model_file:
#     pickle.dump(clf, model_file)

# with open("label_encoder.pkl", "wb") as le_file:
#     pickle.dump(label_encoder, le_file)

# print("✅ Model and label encoder saved as .pkl files.")

# Predict and evaluate
y_pred = clf.predict(X_test)

# # Fix potential mismatch between label_encoder classes and test labels
# unique_labels = sorted(list(set(y_test) | set(y_pred)))
# target_names = label_encoder.inverse_transform(unique_labels)

# # Display evaluation
# print("===== Classification Report =====")
# print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

# print("===== Confusion Matrix =====")
# print(confusion_matrix(y_test, y_pred, labels=unique_labels))

# This is For binary classification
y_pred = clf.predict(X_test)
print("\n===== Classification Report (Binary) =====")
print(classification_report(y_test, y_pred, target_names=["Not Stressed", "Stressed"]))

print("===== Confusion Matrix (Binary) =====")
print(confusion_matrix(y_test, y_pred))
