from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app, origins=["https://rentpricesprediction.vercel.app"])

pipeline = joblib.load("./models/best_model.pkl")

print("=== Model loaded successfully ===")

def predict_price_helper(raw_listing):
    new_data = pd.DataFrame([raw_listing])
    new_data = new_data.rename(columns={
        "is_barrier_free": "is_barrier-free",
        "has_built_in_kitchen": "has_built-in_kitchen"
    })
    prediction = pipeline.predict(new_data)
    return float(prediction[0])

@app.route("/predict", methods=["POST"])
def predict_price_route():
    try:
        data = request.json
        print(f"Received: {data}")
        prediction = predict_price_helper(data)
        print(f"Prediction: {prediction}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
