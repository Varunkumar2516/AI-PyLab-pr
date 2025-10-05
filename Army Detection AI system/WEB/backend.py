#works for smal size vedios 
#consisting tokens less than google/gemini-2.0-flash-exp (that is 1.04 million tokens)
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import cv2
import json
from datetime import datetime
from pathlib import Path
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import mediapipe as mp
from ultralytics import YOLO
import time
import requests

# Constants from two.py
YOLO_MODEL_PATH = "yolov8n.pt"
CLOTH_CLASSIFIER_PATH = "clothing_classifier.pth"
OUTPUT_JSON = "frame_log_with_clothing_and_actions.json"
CONF_THRESH = 0.35
WANTED_OBJECTS = {"person", "car", "laptop", "cell phone", "book", "bottle", "handbag", "gun", "knife", "airpods", "charger"}
WANTED_PERSON_LABELS = {"person"}
CLOTHING_CLASSES = ["army", "doctor", "civilian", "other"]
ACTION_CLASSES = ["running", "crouching", "aiming_weapon", "static"]
SEQ_LENGTH = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANCE_THRESHOLD = 0.01

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Helpers
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_json(p):
    try:
        if Path(p).exists():
            with open(p, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

def save_json(p, data):
    try:
        with open(p, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON: {e}")

def heuristic_classify(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    h_med = int(np.median(h))
    s_med = int(np.median(s))
    if v.mean() > 200 and s.mean() < 40:
        return "doctor", 0.6
    if 40 <= h_med <= 90 and s_med > 50:
        return "army", 0.5
    return "civilian", 0.5

# LSTM for action
class ActionLSTM(nn.Module):
    def __init__(self, input_size=33*3, hidden_size=128, num_classes=4):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Processor class (adapted from VideoProcessor)
class Processor:
    def __init__(self):
        # Load YOLO
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.names_map = self.model.model.names if hasattr(self.model, "model") else self.model.names
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            self.names_map = {}

        # Load clothing classifier
        self.classifier = None
        try:
            clf = models.resnet18(weights=None)
            clf.fc = nn.Linear(clf.fc.in_features, len(CLOTHING_CLASSES))
            clf.load_state_dict(torch.load(CLOTH_CLASSIFIER_PATH, map_location=DEVICE, weights_only=True))
            clf.to(DEVICE).eval()
            self.classifier = clf
        except Exception as e:
            print(f"Using heuristic classifier. Error: {e}")

        self.clf_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load action model
        self.action_model = ActionLSTM().to(DEVICE)
        try:
            self.action_model.load_state_dict(torch.load("action_lstm.pth", map_location=DEVICE, weights_only=True))
        except:
            print("No action model found, using random init for demo.")
        self.action_model.eval()

        # MediaPipe Pose
        self.pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Logging
        self.log_records = load_json(OUTPUT_JSON)
        self.current_record = None
        self.prev_summary = None
        self.frame_idx = 0
        self.lock = threading.Lock()

        # Keypoints queue and EMA
        self.keypoints_queue = queue.deque(maxlen=SEQ_LENGTH)
        self.ema_keypoints = None
        self.ema_alpha = 0.7

    def classify_crop(self, crop):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # Convert to RGB for classifier
        if self.classifier is None:
            return heuristic_classify(crop)
        try:
            img_t = self.clf_transforms(crop_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.classifier(img_t)
                probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
                idx = int(probs.argmax())
                return CLOTHING_CLASSES[idx], float(probs[idx])
        except Exception as e:
            print(f"Classifier error: {e}")
            return heuristic_classify(crop)

    def compute_keypoints_variance(self, keypoints_seq):
        if len(keypoints_seq) < 2:
            return float('inf')
        keypoints_array = np.array(keypoints_seq)
        coords = keypoints_array[:, :-1:3]
        return np.var(coords, axis=0).mean()

    def predict_action(self, keypoints_seq):
        if len(keypoints_seq) < SEQ_LENGTH:
            return "unknown", 0.0
        variance = self.compute_keypoints_variance(keypoints_seq)
        if variance < VARIANCE_THRESHOLD:
            return "static", 0.9
        seq_tensor = torch.tensor([keypoints_seq]).float().to(DEVICE)
        with torch.no_grad():
            output = self.action_model(seq_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            return ACTION_CLASSES[idx], float(probs[idx])

    def process_frame(self, img):
        self.frame_idx += 1

        # YOLO Detection
        results = self.model(img, conf=CONF_THRESH, imgsz=640)
        objects_in_frame = []
        persons_data = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESH:
                    continue
                cls_id = int(box.cls)
                label = self.names_map.get(cls_id, str(cls_id))
                if label not in WANTED_OBJECTS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label in WANTED_PERSON_LABELS:
                    # Crop for clothing and pose
                    pad = int(0.05 * max(x2 - x1, y2 - y1))
                    x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
                    x2c, y2c = min(img.shape[1] - 1, x2 + pad), min(img.shape[0] - 1, y2 + pad)
                    crop = img[y1c:y2c, x1c:x2c].copy()
                    if crop.size == 0:
                        continue

                    # Clothing classification
                    category, score = self.classify_crop(crop)

                    # Pose
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(rgb_crop)
                    if pose_results.pose_landmarks:
                        keypoints = []
                        for lm in pose_results.pose_landmarks.landmark:
                            keypoints.extend([lm.x, lm.y, lm.visibility])
                        if self.ema_keypoints is None:
                            self.ema_keypoints = np.array(keypoints)
                        else:
                            self.ema_keypoints = self.ema_alpha * np.array(keypoints) + (1 - self.ema_alpha) * self.ema_keypoints
                        self.keypoints_queue.append(self.ema_keypoints.tolist())
                        action, action_conf = self.predict_action(list(self.keypoints_queue))
                    else:
                        action, action_conf = "no_pose", 0.0

                    persons_data.append({"category": category, "action": action})

                objects_in_frame.append({"label": label, "conf": conf})

        # Logging
        summary_objects = sorted([f"{o['label']}" for o in objects_in_frame])
        summary_persons = sorted([f"{p['category']}:{p['action']}" for p in persons_data])
        summary = summary_objects + summary_persons

        with self.lock:
            if summary != self.prev_summary:
                if self.current_record:
                    self.current_record["end_time"] = now_iso()
                    self.log_records.append(self.current_record)
                    save_json(OUTPUT_JSON, self.log_records)
                
                self.current_record = {
                    "frame": self.frame_idx,
                    "start_time": now_iso(),
                    "end_time": None,
                    "objects": [o["label"] for o in objects_in_frame],
                    "persons": persons_data
                }

            self.prev_summary = summary

# Function to process video
def process_video(file_path):
    processor = Processor()
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processor.process_frame(frame)
    cap.release()

    # Close current record
    with processor.lock:
        if processor.current_record:
            processor.current_record["end_time"] = now_iso()
            processor.log_records.append(processor.current_record)
            save_json(OUTPUT_JSON, processor.log_records)

latest_sitrep = ""  #Globalvar to store latest SITREP

# Function to generate SITREP using OpenRouter
def generate_sitrep(json_path):
    data = load_json(json_path)
    json_data = json.dumps(data, indent=2)
    date_time = now_iso()

    prompt = f"""Based on the following detection log: {json_data}

Generate a SITREP in the following format:
SITREP – {date_time}
--------------------------------
Situation: {{situation}}
Enemy Activity: {{enemy_activity}}
Human Analysis: {{human_analysis}}
Logistics: {{logistics}}
Command & Signal: {{command_signal}}
"""

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "<YOUR_SITE_URL>",
                    "X-Title": "<YOUR_SITE_NAME>",
                },
                data=json.dumps({
                    "model": "google/gemini-2.0-flash-exp:free",
                    "messages": [{"role": "user", "content": prompt}]
                }),
                timeout=30
            )
            response_json = response.json()
            if response.status_code == 200:
                if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                    return response_json['choices'][0]['message']['content']
                else:
                    print("Missing 'choices' or 'message' in response:", response_json)
                    return f"Unexpected response format: {json.dumps(response_json)}"
            else:
                if 'error' in response_json:
                    error_msg = response_json['error'].get('message', 'Unknown error')
                    print(f"API error (attempt {attempt+1}): {error_msg}")
                    return f"API error: {error_msg}"
                else:
                    print(f"Error generating SITREP (attempt {attempt+1}):", response.text)
                    return f"Error: {response.text}"
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt+1}): {e}")
            return f"Request error: {str(e)}"
        except ValueError as e:  #JSON decoderror
            print(f"JSON decode error (attempt {attempt+1}): {e}")
            return f"JSON decode error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error (attempt {attempt+1}): {e}")
            return f"Unexpected error: {str(e)}"
        time.sleep(5 * (attempt + 1))  #Exponential backoff
    
    return "Error generating SITREP: API may be overloaded or unavailable."

# Periodic SITREP generator
def generate_periodic_sitrep():
    global latest_sitrep
    sitrep_interval = 300  #5 min in sec
    last_sitrep_time = time.time()
    retries = 3
    while True:
        time.sleep(60)  #check every min
        if time.time() - last_sitrep_time > sitrep_interval:
            last_sitrep_time = time.time()
            log_records = load_json(OUTPUT_JSON)
            detected_objects = set()
            detected_movements = set()
            detected_persons = 0
            for record in log_records:
                for obj in record.get('objects', []):
                    detected_objects.add(obj)
                persons = record.get('persons', [])
                detected_persons += len(persons)
                for p in persons:
                    detected_movements.add(p['action'])
            date_time = datetime.now().strftime("%d %b %Y, %H:%M hrs")
            situation = f"Detected objects: {', '.join(detected_objects) if detected_objects else 'None'} near area."
            enemy_activity = f"Detected {detected_persons} individuals; movements: {', '.join(detected_movements) if detected_movements else 'None'}."
            human_analysis = "Emotions not integrated yet."  # Add if needed
            logistics = "No logistics data."
            command_signal = "Comms stable."
            
            sitrep_prompt = f"""
            Generate a SITREP in the following format:
            SITREP – {date_time}
            --------------------------------
            Situation: {situation}
            Enemy Activity: {enemy_activity}
            Human Analysis: {human_analysis}
            Logistics: {logistics}
            Command & Signal: {command_signal}
            """
            
            success = False
            for attempt in range(retries):
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": "Bearer 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",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "<YOUR_SITE_URL>",
                            "X-Title": "<YOUR_SITE_NAME>",
                        },
                        data=json.dumps({
                            "model": "google/gemini-2.0-flash-exp:free",
                            "messages": [{"role": "user", "content": sitrep_prompt}]
                        }),
                        timeout=30  #add timeout to prevent hanging
                    )
                    response_json = response.json()
                    if response.status_code == 200:
                        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                            sitrep_content = response_json['choices'][0]['message']['content']
                            latest_sitrep = sitrep_content
                            print("Periodic SITREP:\n", sitrep_content)
                            success = True
                            break
                        else:
                            print("Missing 'choices' or 'message' in response:", response_json)
                            latest_sitrep = f"Unexpected response format: {json.dumps(response_json)}"
                    else:
                        if 'error' in response_json:
                            error_msg = response_json['error'].get('message', 'Unknown error')
                            print(f"API error (attempt {attempt+1}): {error_msg}")
                            latest_sitrep = f"API error: {error_msg}"
                        else:
                            print(f"Error generating periodic SITREP (attempt {attempt+1}):", response.text)
                            latest_sitrep = f"Error: {response.text}"
                except requests.exceptions.RequestException as e:
                    print(f"Request error (attempt {attempt+1}): {e}")
                    latest_sitrep = f"Request error: {str(e)}"
                except ValueError as e:  # JSON decode error
                    print(f"JSON decode error (attempt {attempt+1}): {e}")
                    latest_sitrep = f"JSON decode error: {str(e)}"
                except Exception as e:
                    print(f"Unexpected error (attempt {attempt+1}): {e}")
                    latest_sitrep = f"Unexpected error: {str(e)}"
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            
            if not success:
                latest_sitrep = "Error generating SITREP: API may be overloaded or unavailable."

# Flask aap
app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.before_request
def add_cors_headers():
    # Add CORS headers to all responses
    if request.method == 'OPTIONS':
        response = app.make_response('')
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    else:
        # For non-OPTIONS requests, set the headers after the request is processed
        pass

@app.after_request
def add_cors_after(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/upload', methods=['POST'])
def upload():
    global latest_sitrep
    if 'video' not in request.files:
        return 'No video file', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            
            # Process the video
            process_video(file_path)
            
            # Generate SITREP using OpenRouter directly
            sitrep = generate_sitrep(OUTPUT_JSON)
            latest_sitrep = sitrep
        except Exception as e:
            latest_sitrep = f"Error processing upload: {str(e)}"
            return f"Error processing upload: {str(e)}", 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return sitrep

@app.route('/get_sitrep', methods=['GET'])
def get_sitrep():
    global latest_sitrep
    return latest_sitrep if latest_sitrep else 'No SITREP available yet.'

if __name__ == '__main__':
    threading.Thread(target=generate_periodic_sitrep, daemon=True).start()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
#   app.run(0.0.0.0, debug=True, port=5000) agar app yeh apne phone mein chaloge toh comment out kro
