import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns


# from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\session_features_30s.csv")

# Binary label: 1 = Stressed, 0 = Not Stressed
df['stress_binary'] = df['stress_label'].apply(lambda x: 1 if 'Stressed' in x else 0)

# Feature selection
feature_columns = [
    'avg_keypress_duration', 'keypress_count', 'backspace_count', 'error_rate',
    'avg_mouse_speed', 'mouse_move_count', 'mouse_click_count',
    'hour', 'day_of_week', 'daylight_morning', 'daylight_evening', 'session_active'
]
X = df[feature_columns]
y = df['stress_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------
# Apply SMOTE to training set only
# ----------------------------
# smote = SMOTE(random_state=42, k_neighbors=1)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ----------------------------
# Scale data for models that require it
# ----------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_resampled)
# X_test_scaled = scaler.transform(X_test)

# Scaling for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\n==== {name} ====")
    if name in ["SVM", "KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        # model.fit(X_train_scaled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        # model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
    
    # if name == "Random Forest":
    #     # Save the model
    #     with open("rf_model_30s.pkl", "wb") as f:
    #         pickle.dump(model, f)

    #     # Save the scaler (used for scaled models)
    #     with open("scaler_30s.pkl", "wb") as f:
    #         pickle.dump(scaler, f)

    #     print("✅ Random Forest model and scaler saved.")
    
    # if name == "Random Forest":
    #     # Confusion Matrix Plot
    #     cm = confusion_matrix(y_test, y_pred)
    #     plt.figure(figsize=(5, 4))
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Stressed", "Stressed"], yticklabels=["Not Stressed", "Stressed"])
    #     plt.title("Confusion Matrix - Random Forest")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.tight_layout()
    #     plt.savefig("confusion_matrix_mouse.png")
    #     plt.show()

    #     # Feature Importance Plot
    #     importances = model.feature_importances_
    #     indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
    #     sorted_features = [feature_columns[i] for i in indices]
    #     sorted_importances = [importances[i] for i in indices]

    #     plt.figure(figsize=(8, 6))
    #     sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    #     plt.title("Feature Importance - Random Forest")
    #     plt.xlabel("Importance")
    #     plt.ylabel("Feature")
    #     plt.tight_layout()
    #     plt.savefig("feature_importance_mouse.png")
    #     plt.show()
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(classification_report(y_test, y_pred, target_names=["Not Stressed", "Stressed"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
# Print the accuracies of all models
print("\nModel Accuracies:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")
    
# # Plot model comparison
# plt.figure(figsize=(8, 5))
# model_names = list(results.keys())
# accuracies = list(results.values())
# sns.barplot(x=accuracies, y=model_names, palette="Blues_d")
# plt.xlabel("Accuracy")
# plt.title("Model Accuracy Comparison - Behavioral Stress Detection")
# plt.xlim(0, 1)
# plt.tight_layout()
# plt.savefig("model_comparison_mouse.png")
# plt.show()
