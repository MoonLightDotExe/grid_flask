from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from google.cloud import vision
import cv2
import os
import numpy as np
import io

app = Flask(__name__)

# Set up Google Vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"  # Update with your key.json path
client = vision.ImageAnnotatorClient()

# Load the TensorFlow model
MODEL_PATH = "models/classifier_new_2.keras"
model = load_model(MODEL_PATH)

# Class labels for the predictions
class_labels = ['fresh_apple', 'stale_apple', 'fresh_banana', 'stale_banana',
                'fresh_tomato', 'stale_tomato', 'fresh_capsicum', 'stale_capsicum',
                'fresh_orange', 'stale_orange']

# Function to preprocess frames and predict
def process_frame(frame):
    try:
        # Resize the frame to match the model input size
        resized_frame = cv2.resize(frame, (224, 224))  
        image = img_to_array(resized_frame) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(image)
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        return predicted_class, confidence
    except Exception as e:
        return "Error", 0

# Video streaming generator
def generate_frames():
    camera = cv2.VideoCapture(0)  # Open webcam (0 = default camera)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Predict for the current frame
            predicted_class, confidence = process_frame(frame)

            # Overlay the prediction on the frame
            text = f"{predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/freshness_classifier')
def freshness_classifier():
    return render_template('freshness_classifier.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= Object Detection =================

@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400

        uploaded_file = request.files['image']
        if uploaded_file.filename == '':
            return "No selected file", 400

        # Save the uploaded file
        input_path = "static/input_image.jpg"
        uploaded_file.save(input_path)

        # Google Vision API for object detection
        with io.open(input_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.object_localization(image=image)
        objects = response.localized_object_annotations

        # Filter out persons and draw bounding boxes
        image_cv = cv2.imread(input_path)
        for obj in objects:
            if obj.name.lower() == 'person' or obj.score < 0.5:
                continue

            vertices = [(vertex.x * image_cv.shape[1], vertex.y * image_cv.shape[0])
                        for vertex in obj.bounding_poly.normalized_vertices]
            vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image_cv, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
            label = f"{obj.name} ({obj.score:.2f})"
            cv2.putText(image_cv, label, (int(vertices[0][0][0]), int(vertices[0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = "static/output_image.jpg"
        cv2.imwrite(output_path, image_cv)

        return render_template('object_detection.html', input_image=input_path, output_image=output_path)
    return render_template('object_detection.html')

if __name__ == '__main__':
    app.run(debug=True)
