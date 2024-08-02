import os
import cv2
import awsgi
import base64
import numpy as np
import moviepy.editor as mp
from ultralytics import YOLO
from PIL import Image as PILImage
from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

model_path = r'models/best.pt'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file does not exist at the specified path: {model_path}")

model = YOLO(model_path)

TARGET_CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Safety Vest']
CLASS_COLORS = {
    'Hardhat': (0, 255, 0),
    'Mask': (255, 0, 0),
    'NO-Hardhat': (0, 0, 255),
    'NO-Mask': (255, 255, 0),
    'NO-Safety Vest': (0, 255, 255),
    'Safety Vest': (255, 0, 255)
}

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
            os.remove(filename)  
            return redirect(url_for('display_video', filename=processed_filename))
        else:
            return display_image(file)

def display_image(file):
    image = PILImage.open(file).convert('RGB')
    image_np = np.array(image)
    results = model.predict(source=image_np)
    pred_image = customize_results(image_np, results)
    
    _, buffer = cv2.imencode('.jpg', pred_image)
    pred_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return render_template('display_image.html', image_data=pred_image_base64)

def process_frame(frame):
    results = model.predict(source=frame)
    pred_frame = customize_results(frame, results)
    return pred_frame

def customize_results(frame, results):
    frame = frame.copy() 
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            if class_name in TARGET_CLASSES:
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                confidence = float(box.conf)  
                label = f"{class_name} {confidence:.2f}"
                xyxy = box.xyxy[0]  
                x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

def process_video(filepath):
    input_path = filepath
    output_filename = 'processed_' + os.path.basename(filepath)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    video = mp.VideoFileClip(input_path)
    processed_video = video.fl_image(process_frame)
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

@app.route('/upload_another_video', methods=['GET'])
def upload_another_video():
    for filename in os.listdir(app.config['OUTPUT_FOLDER']):
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    return redirect(url_for('index'))

def handler(event, context):
    return awsgi.response(app, event, context)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)