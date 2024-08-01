from flask import Flask, request, render_template, send_file, redirect, url_for, Response
import cv2
from ultralytics import YOLO
from PIL import Image as PILImage
import numpy as np
import os
import base64
import logging
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
            return redirect(url_for('display_video', filename=file.filename))
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

@app.route('/display_video/<filename>')
def display_video(filename):
    return render_template('display_video.html', filename=filename)
    # filename = request.args.get('filename')
    # input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_' + filename)

    # if not os.path.exists(input_path):
    #     return 'File not found', 404

    # logging.debug(f"Processing video: {input_path}")
    # process_video(input_path, output_path)

    # if os.path.exists(input_path):
    #     os.remove(input_path)  # Delete the original video after processing

    # if not os.path.exists(output_path):
    #     logging.error(f"Processed video not found: {output_path}")
    #     return 'Processed video not found', 404

    # logging.debug(f"Processed video saved at: {output_path}")
    # return render_template('display_video.html', video_filename='processed_' + filename)

# def process_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = model.predict(source=rgb_frame)
#         pred_image = results[0].plot()
#         out.write(pred_image)
    
#     cap.release()
#     out.release()

# @app.route('/video/<filename>')
# def serve_video(filename):
#     video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
#     if not os.path.exists(video_path):
#         logging.error(f"Video not found: {video_path}")
#         return 'Video not found', 404
#     logging.debug(f"Serving video: {video_path}")
#     return send_file(video_path, as_attachment=False, mimetype='video/mp4')

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

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = file.filename
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         return redirect(url_for('play_video', filename=filename))
#     return redirect(request.url)

