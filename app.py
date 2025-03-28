from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Flask app initialization
app = Flask(__name__)

# Load the .h5 model
model = tf.keras.models.load_model("model.h5")

@app.route("/")
def home():
    return "Model Deployment Successful!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Reshape for model

        # Make prediction
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

