from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import base64
from collections import defaultdict
import os
import tempfile

app = Flask(__name__)

# Define supported animals based on model capabilities
SUPPORTED_ANIMALS = {
    'person': 'person',
    'bird': 'wild',
    'cat': 'pet',
    'dog': 'pet',
    'horse': 'pet',
    'sheep': 'wild',
    'cow': 'pet',
    'elephant': 'wild',
    'bear': 'wild',
    'zebra': 'wild',
    'giraffe': 'wild'
}

# Cache the model to avoid reloading
model = None

def get_model():
    """Get or initialize YOLOv8 model"""
    global model
    if model is None:
        model = YOLO('yolov8x.pt')
    return model

def process_detection(frame, detection, class_name, conf_threshold=0.3):
    """Process single detection and return detection info"""
    if len(detection) < 6:  # Ensure detection has enough elements
        return None
        
    x1, y1, x2, y2, conf = detection[:5]
    
    if conf < conf_threshold:
        return None
    
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    if (x2 - x1) * (y2 - y1) < 100:  # Filter out too small detections
        return None
    
    animal_type = SUPPORTED_ANIMALS.get(class_name.lower(), 'unknown')
    
    # Color mapping
    color_map = {
        'pet': (0, 255, 0),    # Green
        'wild': (0, 0, 255),   # Red
        'farm': (255, 165, 0), # Orange
        'person': (255, 0, 255) # Purple
    }
    
    color = color_map.get(animal_type, (128, 128, 128))
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Add label with confidence
    label = f"{class_name} ({conf:.2f})"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return {
        'class': class_name,
        'type': animal_type,
        'confidence': float(conf),
        'bbox': [int(x1), int(y1), int(x2), int(y2)]
    }

def process_image(image_array, conf_threshold=0.3):
    """Process image and return detections"""
    model = get_model()
    results = model(image_array, conf=conf_threshold)[0]
    detections = []
    dominant_type = defaultdict(float)  # Using float for confidence-weighted counting
    
    for r in results.boxes.data.tolist():
        class_name = results.names[int(r[5])]
        if class_name.lower() in SUPPORTED_ANIMALS:
            detection = process_detection(image_array, r, class_name, conf_threshold)
            if detection:
                detections.append(detection)
                # Weight by confidence
                dominant_type[detection['type']] += detection['confidence']
    
    # Determine the dominant category
    detected_type = max(dominant_type.items(), key=lambda x: x[1])[0] if dominant_type else "unknown"
    return detections, detected_type, image_array

def process_video(video_path, conf_threshold=0.3):
    """Process video file and save output"""
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Create temporary file for output
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
    
    detections_summary = defaultdict(float)  # Using float for confidence-weighted counting
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, conf=conf_threshold)[0]
        
        for r in results.boxes.data.tolist():
            class_name = results.names[int(r[5])]
            if class_name.lower() in SUPPORTED_ANIMALS:
                detection = process_detection(frame, r, class_name, conf_threshold)
                if detection:
                    # Weight by confidence
                    detections_summary[detection['type']] += detection['confidence']
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Convert to MP4 container using FFmpeg if available
    try:
        final_output = temp_output.replace('.mp4', '_final.mp4')
        os.system(f"ffmpeg -i {temp_output} -vcodec libx264 {final_output}")
        os.remove(temp_output)
        temp_output = final_output
    except:
        pass
    
    # Determine the dominant category based on confidence-weighted detections
    detected_type = max(detections_summary.items(), key=lambda x: x[1])[0] if detections_summary else "unknown"
    
    return temp_output, dict(detections_summary), detected_type

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    file_type = request.form.get('type', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        if file_type == 'image':
            # Process image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            detections, detected_type, processed_img = process_image(img)
            
            # Convert processed image to base64
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'detections': detections,
                'dominant_type': detected_type,
                'processed_image': f'data:image/jpeg;base64,{img_base64}'
            })
            
        elif file_type == 'video':
            # Save uploaded video to temporary file
            temp_input = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            file.save(temp_input)
            
            # Process video
            try:
                output_path, detections_summary, detected_type = process_video(temp_input)
                
                # Convert output video to base64
                with open(output_path, 'rb') as video_file:
                    video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                
                # Clean up temporary files
                os.remove(temp_input)
                os.remove(output_path)
                
                return jsonify({
                    'detections': detections_summary,
                    'dominant_type': detected_type,
                    'processed_video': f'data:video/mp4;base64,{video_base64}'
                })
                
            except Exception as e:
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise e
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)