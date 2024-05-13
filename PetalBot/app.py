from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from PIL import Image
import numpy as np
from io import BytesIO
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

# Load the pre-trained model for Iris species classification
iris_model = keras.models.load_model(r'D:\UNIVERSITY\11-SPRING 24\FYP-II\PetalBot\Iris Detection Models\model250.h5')

# Load the pre-trained model for flower detection
flower_model = keras.models.load_model(r'D:\UNIVERSITY\11-SPRING 24\FYP-II\PetalBot\Flower detection Models\flowerDetectionmodel160.h5')

# Define labels for flower detection model
flower_labels = ['animal', 'flower', 'human', 'objects']

def contains_flower(image):
    # Check if a file has been provided
    if image is None:
        return False

    try:
        # Read the image using PIL
        img = Image.open(image)
        img = img.resize((224, 224))  # Resize the image to match the model's input size
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Make a prediction using the flower detection model
        prediction = flower_model.predict(np.array([img_array]))
        predicted_class_index = np.argmax(prediction)
        
        # Check if the predicted class is "flower"
        return flower_labels[predicted_class_index] == 'flower'
    except Exception as e:
        print(f"Error detecting flower:")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        print(request.files)  # Log the files in the console
        uploaded_image = request.files['image']
        # Perform flower detection
        if contains_flower(uploaded_image):
            # If the image contains a flower, proceed with Iris species classification
            img = Image.open(uploaded_image)
            img = img.resize((224, 224))  # Resize the image to match the model's input size
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize pixel values
        
            # Make a prediction using the Iris species classification model
            prediction = iris_model.predict(np.array([img_array]))

            # Convert the prediction to a class label
            iris_labels = ['iris-setosa', 'iris-versicolour', 'iris-virginica']
            predicted_class = iris_labels[np.argmax(prediction)]
            return jsonify({'predicted_class': predicted_class})
        else:
            # If the image does not contain a flower, return a different response
            return jsonify({'message': 'The uploaded image does not contain a flower.'})
            
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
