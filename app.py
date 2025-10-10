import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import logging
import gc

# Configure TensorFlow to use less memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging

tensorflow_available = False
tf_module = None
cv2_module = None

try:
    import tensorflow as tf
    tf_module = tf
    tensorflow_available = True
    print("TensorFlow loaded successfully")
    # Configure TensorFlow to use less memory
    try:
        if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
    except Exception as e:
        print(f"Warning: Could not configure TensorFlow memory: {e}")
except ImportError as e:
    tf_module = None
    print(f"Warning: TensorFlow not available: {e}")

try:
    import cv2
    cv2_module = cv2
    print("OpenCV loaded successfully")
except ImportError:
    try:
        import cv2.cv2 as cv2
        cv2_module = cv2
        print("OpenCV cv2.cv2 loaded successfully")
    except ImportError:
        cv2_module = None
        print("Warning: OpenCV not available. Face detection will be disabled. This is normal in containerized environments.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # Reduced from 10MB to 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
face_cascade = None
model_loaded = False
cascade_loaded = False

def load_model_and_cascade():
    global model, face_cascade, model_loaded, cascade_loaded
    
    if not tensorflow_available:
        logger.warning("TensorFlow not available. Using fallback mode.")
        model_loaded = False
        cascade_loaded = False if cv2_module is None else True
        return
    
    try:
        model_path = 'best_mask_model.h5'
        if os.path.exists(model_path):
            try:
                # Load model with memory optimization
                if tf_module is not None:
                    # Configure TensorFlow to use less memory
                    tf_module.config.experimental.set_memory_limit(1024)  # Limit to 1GB
                    model = tf_module.keras.models.load_model(model_path)
                    model_loaded = True
                    logger.info("Face mask detection model loaded successfully")
                else:
                    model_loaded = False
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                model_loaded = False
        else:
            logger.error(f"Model file not found: {model_path}")
            model_loaded = False
            
        cascade_path = 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path) and cv2_module is not None:
            face_cascade = cv2_module.CascadeClassifier(cascade_path)
            cascade_loaded = True
            logger.info("Face cascade classifier loaded successfully")
        elif cv2_module is None:
            logger.warning("OpenCV not available. Face detection disabled.")
            cascade_loaded = False
        else:
            logger.error(f"Cascade file not found: {cascade_path}")
            cascade_loaded = False
            
    except Exception as e:
        logger.error(f"Error loading model or cascade: {str(e)}")
        model_loaded = False
        cascade_loaded = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if cv2_module is not None and tf_module is not None:
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou

    def non_max_suppression(boxes, scores=None, overlap_threshold=0.3):
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        
        if scores is None:
            scores = boxes[:, 2] * boxes[:, 3]
        
        idxs = np.argsort(scores)[::-1]
        
        keep = []
        while len(idxs) > 0:
            current = idxs[0]
            keep.append(current)
            
            ious = []
            for i in idxs[1:]:
                iou = calculate_iou(boxes[current], boxes[i])
                ious.append(iou)
            
            ious = np.array(ious)
            suppressed_idx = np.where(ious <= overlap_threshold)[0]
            
            idxs = idxs[1:][suppressed_idx]
        
        return boxes[keep].astype("int")

    def filter_faces_by_size_and_position(faces, image_shape, min_size_ratio=0.05, max_size_ratio=0.8):
        if len(faces) == 0:
            return []
        
        img_h, img_w = image_shape[:2]
        min_size = int(min(img_w, img_h) * min_size_ratio)
        max_size = int(min(img_w, img_h) * max_size_ratio)
        
        filtered_faces = []
        for (x, y, w, h) in faces:
            if w < min_size or h < min_size or w > max_size or h > max_size:
                continue
                
            face_center_x = x + w/2
            face_center_y = y + h/2
            
            if (face_center_x < img_w * 0.1 or face_center_x > img_w * 0.9 or
                face_center_y < img_h * 0.1 or face_center_y > img_h * 0.9):
                continue
                
            filtered_faces.append([x, y, w, h])
        
        return np.array(filtered_faces)

    def detect_faces_and_masks(image):
        global model, face_cascade
        
        if not tensorflow_available or model is None or face_cascade is None:
            return []
        
        try:
            gray = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(40, 40),
                flags=cv2_module.CASCADE_SCALE_IMAGE
            )
            
            faces = filter_faces_by_size_and_position(faces, image.shape, min_size_ratio=0.05, max_size_ratio=0.7)
            
            if len(faces) > 1:
                faces = non_max_suppression(faces, overlap_threshold=0.2)
            
            detections = []
            
            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                
                face_img_resized = cv2_module.resize(face_img, (224, 224))
                face_img_array = tf_module.keras.utils.img_to_array(face_img_resized)
                face_img_array = tf_module.expand_dims(face_img_array, 0)
                face_img_array /= 255.0
                
                prediction = model.predict(face_img_array, verbose=0)
                
                mask_probability = float(1 - prediction[0][0])
                no_mask_probability = float(prediction[0][0])
                
                if mask_probability > 0.5:
                    label = "Masked"
                    confidence = mask_probability * 100
                else:
                    label = "No Mask"
                    confidence = no_mask_probability * 100
                
                detections.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'label': label,
                    'confidence': confidence
                })
                
            return detections
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []

    def process_image_for_display(image, detections):
        image_copy = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            color = (0, 255, 0) if label == "Masked" else (0, 0, 255)
            
            cv2_module.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)
            
            label_text = f"{label}: {confidence:.1f}%"
            cv2_module.putText(image_copy, label_text, (x, y - 10), 
                           cv2_module.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image_copy
else:
    def detect_faces_and_masks(image):
        return []

    def process_image_for_display(image, detections):
        return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_status')
def model_status():
    return jsonify({
        'model_loaded': model_loaded,
        'cascade_loaded': cascade_loaded,
        'model_loaded_status': 'success' if model_loaded else 'failed',
        'cascade_loaded_status': 'success' if cascade_loaded else 'failed'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
        # Check file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        
        if file_length > MAX_FILE_SIZE:
            return jsonify({'success': False, 'error': 'File too large'}), 400
            
        file_bytes = np.frombuffer(file.read(), np.uint8)
        if cv2_module is not None:
            image = cv2_module.imdecode(file_bytes, cv2_module.IMREAD_COLOR)
        else:
            image = tf_module.io.decode_image(file_bytes, channels=3)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
            
        detections = detect_faces_and_masks(image)
        
        processed_image = process_image_for_display(image, detections)
        
        if cv2_module is not None:
            _, buffer = cv2_module.imencode('.jpg', processed_image, [int(cv2_module.IMWRITE_JPEG_QUALITY), 85])
            img_str = base64.b64encode(buffer).decode()
        else:
            _, img_bytes = tf_module.io.encode_jpeg(processed_image).numpy()
            img_str = base64.b64encode(img_bytes).decode()
        
        # Force garbage collection
        try:
            del image
            del file_bytes
        except:
            pass
        gc.collect()
        
        return jsonify({
            'success': True,
            'detections': detections,
            'image': f'data:image/jpeg;base64,{img_str}'
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        # Force garbage collection on error
        gc.collect()
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Optimized version that uses less memory"""
    try:
        if 'frame' not in request.files:
            return jsonify({'success': False, 'error': 'No frame provided'}), 400
            
        file = request.files['frame']
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        
        if file_length > 2 * 1024 * 1024:  # 2MB limit
            return jsonify({'success': False, 'error': 'Frame too large'}), 400
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        if cv2_module is not None:
            image = cv2_module.imdecode(file_bytes, cv2_module.IMREAD_COLOR)
        else:
            image = tf_module.io.decode_image(file_bytes, channels=3)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid frame'}), 400
            
        detections = detect_faces_and_masks(image)
        
        # Force garbage collection to free memory
        try:
            del image
            del file_bytes
        except:
            pass
        gc.collect()
        
        return jsonify({
            'success': True,
            'detections': detections
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        # Force garbage collection on error
        gc.collect()
        return jsonify({'success': False, 'error': 'Frame processing failed'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'cascade_loaded': cascade_loaded,
        'model_loaded_status': 'success' if model_loaded else 'failed',
        'cascade_loaded_status': 'success' if cascade_loaded else 'failed'
    })

if __name__ == '__main__':
    load_model_and_cascade()
    
    # For Render deployment, we don't want debug mode
    if os.environ.get('RENDER'):
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)