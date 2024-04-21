# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__, static_folder='static', static_url_path='/static')

model = tf.keras.models.load_model("mnist_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # print(data)
        image_data = np.array(data['imageData'])
        test = cv2.resize(image_data, (28, 28))
        test = np.expand_dims(test,axis=0)
        # Print the received image data for debugging
        # print("Received Image Data:", image_data)

        # image_array = np.array([np.array(image_data) / 255.0])
        
        predictions = model.predict(test)

        # Print the predictions for debugging
        # predictions = 0
        # print("Predictions:", predictions)
        # print("Predictions Shape:", predictions.shape)

        predicted_class = np.argmax(predictions[0])

        # Print the predicted class for debugging
        # print("Predicted Class:", predicted_class)

        return jsonify({'result': str(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
