from flask import Flask, request, render_template, redirect, url_for, Response
import moviepy.editor as mp
from ultralytics import YOLO
from PIL import Image as PILImage
import numpy as np
import os
import base64
import logging
import cv2
from flask_cors import CORS

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Path to the trained YOLOv8 model weights
model_path = r'models/best.pt'  # Update the path if different

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file does not exist at the specified path: {model_path}")

# Load the trained YOLOv8 model
model = YOLO(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        if file_ext in ['mp4']:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            processed_filename = process_video(filename)
            return redirect(url_for('display_video', filename=processed_filename))
        else:
            return display_image(file)

def display_image(file):
    # Process image
    image = PILImage.open(file).convert('RGB')
    image_np = np.array(image)
    results = model.predict(source=image_np)
    pred_image = results[0].plot()
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', pred_image)
    pred_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return render_template('display_image.html', image_data=pred_image_base64)

def process_video(filepath):
    input_path = filepath
    output_filename = 'processed_' + os.path.basename(filepath)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Open the video file using moviepy
    video = mp.VideoFileClip(input_path)

    def process_frame(frame):
        # Process each frame using YOLO
        results = model.predict(source=frame)
        pred_frame = results[0].plot()
        return pred_frame

    # Process the video frames
    processed_video = video.fl_image(process_frame)

    # Write the processed video to the output path
    processed_video.write_videofile(output_path, codec='libx264')

    return output_filename

@app.route('/display_video/<filename>')
def display_video(filename):
    return render_template('display_video.html', filename=filename)

@app.route('/cleanup/<filename>')
def cleanup(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)