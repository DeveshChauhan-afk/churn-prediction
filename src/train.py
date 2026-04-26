from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from preprocess import load_data
import pandas as pd
from sklearn.metrics import classification_report

X, y = load_data("D:/New folder/churn/data/churn.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

y_probs = rf.predict_proba(X_test)[:,1]
y_pred_custom = (y_probs > 0.3).astype(int)

print("feature importance\n",feature_importance.sort_values(ascending=False).head(10))
print("confusion matrix\n",confusion_matrix(y_test, rf.predict(X_test)))
# Evaluation
print("LR ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))
print("RF ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
print("classification report")
print(classification_report(y_test, rf.predict(X_test)))
print("custom y pred classification report")
print(classification_report(y_test, y_pred_custom))
# Save best model
joblib.dump(rf, "D:/New folder/churn/model/model.pkl")
joblib.dump(X.columns, "D:/New folder/churn/model/columns.pkl")