import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model, scaler, label encoder, and feature names
with open('d:/HAND_GESTURE/trained_models.pkl', 'rb') as f:
    model_data = pickle.load(f)
models = model_data['models']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']

# Use the best model (SVM or Random Forest)
best_model = models['SVM'] if 'SVM' in models else list(models.values())[0]

# Feature extraction function (must match your training pipeline)
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
    # 1. Distance features
    for tip_idx in fingertips:
        features.append(calculate_distance(wrist, landmarks[tip_idx]))
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            features.append(calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[j]]))
    for mcp_idx in mcps:
        features.append(calculate_distance(wrist, landmarks[mcp_idx]))
    # 2. Finger length features
    for finger in finger_segments:
        total_length = 0
        for i in range(len(finger)-1):
            total_length += calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]])
        features.append(total_length)
        for i in range(len(finger)-1):
            features.append(calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]]))
    # 3. Angle features
    for finger in finger_segments:
        for i in range(1, len(finger)-1):
            features.append(calculate_angle(landmarks[finger[i-1]], landmarks[finger[i]], landmarks[finger[i+1]]))
    # 4. Inter-finger angles
    features.append(calculate_angle(landmarks[4], landmarks[0], landmarks[8]))
    features.append(calculate_angle(landmarks[8], landmarks[0], landmarks[12]))
    features.append(calculate_angle(landmarks[12], landmarks[0], landmarks[16]))
    features.append(calculate_angle(landmarks[16], landmarks[0], landmarks[20]))
    # 5. Palm-related features
    palm_points = [0, 1, 5, 9, 13, 17]
    palm_center = np.mean([landmarks[i] for i in palm_points], axis=0)
    for tip_idx in fingertips:
        features.append(calculate_distance(palm_center, landmarks[tip_idx]))
    # 6. Statistical features
    tip_distances = [calculate_distance(wrist, landmarks[tip_idx]) for tip_idx in fingertips]
    features.append(np.var(tip_distances))
    features.append(np.mean(tip_distances))
    features.append(np.std(tip_distances))
    # 7. Hand orientation features
    wrist_to_middle = np.array(landmarks[9]) - np.array(wrist)
    features.append(np.arctan2(wrist_to_middle[1], wrist_to_middle[0]))
    features.append(np.arctan2(wrist_to_middle[2], np.sqrt(wrist_to_middle[0]**2 + wrist_to_middle[1]**2)))
    # 8. Spread features
    max_spread = 0
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            dist = calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[j]])
            max_spread = max(max_spread, dist)
    features.append(max_spread)
    features.append(calculate_distance(landmarks[4], landmarks[20]))
    # 9. Height features
    z_coords = [landmarks[i][2] for i in range(21)]
    features.append(np.mean(z_coords))
    features.append(np.std(z_coords))
    features.append(max(z_coords) - min(z_coords))
    return np.array(features).reshape(1, -1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    gesture = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmarks
            lm = []
            h, w, _ = frame.shape
            for i in range(21):
                x = hand_landmarks.landmark[i].x * w
                y = hand_landmarks.landmark[i].y * h
                z = hand_landmarks.landmark[i].z * w  # z is relative to width
                lm.append([x, y, z])
            # Extract features and predict
            feats = extract_features(lm)
            import pandas as pd
            feats_df = pd.DataFrame(feats, columns=feature_names)
            feats_scaled = scaler.transform(feats_df)
            pred = best_model.predict(feats_scaled)[0]
            gesture = label_encoder.classes_[pred]
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Show prediction
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
