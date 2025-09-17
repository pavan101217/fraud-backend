import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset
df = pd.read_csv("../transactionscopy.csv")  # adjust path if needed

# 2. Split features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

# 5. Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc
    }
    
    print(f"\n🔹 {name} Results 🔹")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

# 6. Select the best model (by Recall, then F1, then ROC-AUC)
best_model = max(results.items(), key=lambda x: (x[1]["Recall"], x[1]["F1 Score"], x[1]["ROC-AUC"]))
best_model_name = best_model[0]

print("\n✅ Best Model Selected:", best_model_name)

# 7. Retrain the best model on full dataset
final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", models[best_model_name])
])
final_pipeline.fit(X, y)

# 8. Save the best model
joblib.dump(final_pipeline, "fraud_model.joblib")
print(f"\n💾 Saved {best_model_name} as fraud_model.joblib")
