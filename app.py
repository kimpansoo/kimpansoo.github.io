import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # CORS 지원 추가
import threading
import time

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 접근 허용
socketio = SocketIO(app, cors_allowed_origins="*")  # CORS 설정

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sensitivity = 0.16
        self.drowsy_count = 0
        self.max_drowsy_count = 10
        self.camera = None

    def initialize_camera(self):
        """Initialize camera with multiple indices"""
        camera_indices = [0, 1, 2]
        for index in camera_indices:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                self.camera = cap
                return True
        return False

    def calculate_ear(self, landmarks, eye_indices):
        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        vertical1 = distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        vertical2 = distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        horizontal = distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

        return (vertical1 + vertical2) / (2 * horizontal)

    def detect_drowsiness(self):
        if not self.camera:
            self.initialize_camera()

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            is_drowsy = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                                 for lm in face_landmarks.landmark]

                    left_eye = self.calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                    right_eye = self.calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                    ear = (left_eye + right_eye) / 2.0

                    if ear < self.sensitivity:
                        self.drowsy_count += 1
                        is_drowsy = self.drowsy_count > self.max_drowsy_count
                    else:
                        self.drowsy_count = max(0, self.drowsy_count - 1)

            if is_drowsy:
                cv2.putText(frame, "DROWSY!!!", (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

            # Convert frame to base64 for WebSocket transmission
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'frame': frame_base64, 'drowsy': is_drowsy})

            time.sleep(0.03)  # ~30 fps

    def start_detection(self):
        detection_thread = threading.Thread(target=self.detect_drowsiness)
        detection_thread.daemon = True
        detection_thread.start()

drowsiness_detector = DrowsinessDetector()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    drowsiness_detector.start_detection()

@socketio.on('update_sensitivity')
def update_sensitivity(data):
    drowsiness_detector.sensitivity = float(data['sensitivity'])
    emit('sensitivity_updated', {'sensitivity': drowsiness_detector.sensitivity})

if __name__ == '__main__':
    # 모든 네트워크 인터페이스에서 접근 가능하도록 설정
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
