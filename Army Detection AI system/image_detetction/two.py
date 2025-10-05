import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from datetime import datetime
from pathlib import Path
import json
import threading
import queue

# Constants
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
    return json.loads(Path(p).read_text()) if Path(p).exists() else []

def save_json(p, data):
    Path(p).write_text(json.dumps(data, indent=2))

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

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Load YOLO
        self.model = YOLO(YOLO_MODEL_PATH)
        self.names_map = self.model.model.names if hasattr(self.model, "model") else self.model.names
        print(f"Loaded YOLO model with classes: {list(self.names_map.values())}")

        # Load clothing classifier
        self.classifier = None
        try:
            clf = models.resnet18(weights=None)
            clf.fc = nn.Linear(clf.fc.in_features, len(CLOTHING_CLASSES))
            clf.load_state_dict(torch.load(CLOTH_CLASSIFIER_PATH, map_location=DEVICE, weights_only=True))
            clf.to(DEVICE).eval()
            self.classifier = clf
            print("Loaded clothing classifier.")
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
            print("Loaded action LSTM.")
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
        if self.classifier is None:
            return heuristic_classify(crop)
        try:
            img_t = self.clf_transforms(crop).unsqueeze(0).to(DEVICE)
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

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
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

                # Draw bounding box and label for all wanted objects
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Detected: {label} with confidence {conf:.2f}")

                if label in WANTED_PERSON_LABELS:
                    # Crop for clothing and pose
                    pad = int(0.05 * max(x2 - x1, y2 - y1))
                    x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
                    x2c, y2c = min(img.shape[1] - 1, x2 + pad), min(img.shape[0] - 1, y2 + pad)
                    crop = img[y1c:y2c, x1c:x2c].copy()
                    if crop.size == 0:
                        print("Empty crop, skipping person processing")
                        continue

                    # Clothing classification
                    category, score = self.classify_crop(crop)
                    print(f"Clothing: {category} with confidence {score:.2f}")

                    # Pose
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(rgb_crop)
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        keypoints = []
                        for lm in pose_results.pose_landmarks.landmark:
                            keypoints.extend([lm.x, lm.y, lm.visibility])
                        if self.ema_keypoints is None:
                            self.ema_keypoints = np.array(keypoints)
                        else:
                            self.ema_keypoints = self.ema_alpha * np.array(keypoints) + (1 - self.ema_alpha) * self.ema_keypoints
                        self.keypoints_queue.append(self.ema_keypoints.tolist())
                        action, action_conf = self.predict_action(list(self.keypoints_queue))
                        print(f"Action: {action} with confidence {action_conf:.2f}")
                    else:
                        action, action_conf = "no_pose", 0.0
                        print("No pose detected")

                    # Update main image with crop
                    img[y1c:y2c, x1c:x2c] = crop
                    cv2.putText(img, f"Clothing: {category} {score:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(img, f"Action: {action} {action_conf:.2f}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
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
                print(f"Frame {self.frame_idx}: Objects={summary_objects}, Persons={summary_persons}")

            self.prev_summary = summary

        cv2.putText(img, now_iso(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit App
st.title("Integrated Object, Clothing, Pose, and Action Detector")

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="integrated_detector",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)

st.write("Press 'Start' to begin live detection.")
st.write("Logs saved to: frame_log_with_clothing_and_actions.json")
st.write("Note: Ensure yolov8n.pt and clothing_classifier.pth are in the directory. Retrain action_lstm.pth with static pose data for better action accuracy.")