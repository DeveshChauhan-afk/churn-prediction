from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "Churn Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    df = pd.get_dummies(df)
    prediction = model.predict(df)[0]

    return jsonify({"churn": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)