from flask import Flask, request, jsonify
import joblib
import pandas as pd
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Load once (GLOBAL SCOPE)
model = joblib.load("model/model.pkl")
model_columns = joblib.load("model/columns.pkl")
app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "Churn Prediction API Running"

@app.route("/predict", methods=["POST","GET"])
def predict():

    if request.method == "GET":
        return jsonify({
            "message": "Use POST request with JSON data to get prediction"
        })

    data = request.json

    if not data:
        return jsonify({"error": "Invalid or missing JSON input"}), 400

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    risk = "High Risk" if probability > 0.5 else "Low Risk"

    # Business action
    if probability > 0.7:
        action = "Immediate retention call"
        risk = "High Risk"
    elif probability > 0.4:
        risk = "Medium Risk"
        action = "Offer discount"
    else:
        risk = "Low Risk"
        action = "No action needed"

    return jsonify({
        "churn": int(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": risk,
        "recommended_action": action
    })

if __name__ == "__main__":
    app.run(debug=True)