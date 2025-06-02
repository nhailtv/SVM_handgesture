import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model
with open('d:/HAND_GESTURE/trained_models.pkl', 'rb') as f:
    model_data = pickle.load(f)
models = model_data['models']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']
best_model = models['SVM'] if 'SVM' in models else list(models.values())[0]

# Feature extraction (giống như khi train)
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)

def extract_features(landmarks):
    features = []
    wrist = landmarks[0]
    fingertips = [4, 8, 12, 16, 20]
    mcps = [1, 5, 9, 13, 17]
    finger_segments = [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
        [13, 14, 15, 16], [17, 18, 19, 20]
    ]
    for tip_idx in fingertips:
        features.append(calculate_distance(wrist, landmarks[tip_idx]))
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            features.append(calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[j]]))
    for mcp_idx in mcps:
        features.append(calculate_distance(wrist, landmarks[mcp_idx]))
    for finger in finger_segments:
        total_length = 0
        for i in range(len(finger)-1):
            total_length += calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]])
        features.append(total_length)
        for i in range(len(finger)-1):
            features.append(calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]]))
    for finger in finger_segments:
        for i in range(1, len(finger)-1):
            features.append(calculate_angle(landmarks[finger[i-1]], landmarks[finger[i]], landmarks[finger[i+1]]))
    features.append(calculate_angle(landmarks[4], landmarks[0], landmarks[8]))
    features.append(calculate_angle(landmarks[8], landmarks[0], landmarks[12]))
    features.append(calculate_angle(landmarks[12], landmarks[0], landmarks[16]))
    features.append(calculate_angle(landmarks[16], landmarks[0], landmarks[20]))
    palm_points = [0, 1, 5, 9, 13, 17]
    palm_center = np.mean([landmarks[i] for i in palm_points], axis=0)
    for tip_idx in fingertips:
        features.append(calculate_distance(palm_center, landmarks[tip_idx]))
    tip_distances = [calculate_distance(wrist, landmarks[tip_idx]) for tip_idx in fingertips]
    features.append(np.var(tip_distances))
    features.append(np.mean(tip_distances))
    features.append(np.std(tip_distances))
    wrist_to_middle = np.array(landmarks[9]) - np.array(wrist)
    features.append(np.arctan2(wrist_to_middle[1], wrist_to_middle[0]))
    features.append(np.arctan2(wrist_to_middle[2], np.sqrt(wrist_to_middle[0]**2 + wrist_to_middle[1]**2)))
    max_spread = 0
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            dist = calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[j]])
            max_spread = max(max_spread, dist)
    features.append(max_spread)
    features.append(calculate_distance(landmarks[4], landmarks[20]))
    z_coords = [landmarks[i][2] for i in range(21)]
    features.append(np.mean(z_coords))
    features.append(np.std(z_coords))
    features.append(max(z_coords) - min(z_coords))
    return np.array(features)

# Đánh giá trên folder hagrid_test
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

test_folder = 'd:/HAND_GESTURE/Data/hagrid_test'
X_test, y_test, y_true_labels, y_pred_labels = [], [], [], []

for gesture in os.listdir(test_folder):
    gesture_path = os.path.join(test_folder, gesture)
    if not os.path.isdir(gesture_path):
        continue
    for img_name in os.listdir(gesture_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(gesture_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            lm = []
            for i in range(21):
                x = hand_landmarks.landmark[i].x * w
                y = hand_landmarks.landmark[i].y * h
                z = hand_landmarks.landmark[i].z * w
                lm.append([x, y, z])
            feats = extract_features(lm)
            X_test.append(feats)
            y_test.append(gesture)
        # Nếu không detect được landmark thì bỏ qua ảnh này

if not X_test:
    print('No valid hand landmarks detected in test set.')
    exit()

X_test = np.array(X_test)
y_test_encoded = label_encoder.transform(y_test)
feats_df = pd.DataFrame(X_test, columns=feature_names)
X_test_scaled = scaler.transform(feats_df)
y_pred = best_model.predict(X_test_scaled)

print('Accuracy:', accuracy_score(y_test_encoded, y_pred))
print('Classification Report:')
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
print('Confusion Matrix:')
print(confusion_matrix(y_test_encoded, y_pred))