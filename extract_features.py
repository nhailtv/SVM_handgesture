import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by vectors p1->p2 and p2->p3"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    # Avoid division by zero
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return math.acos(cos_angle)

def extract_features(row):
    """Extract features from a single row of landmark data"""
    # Parse landmarks from the row (21 points, each with x,y,z)
    landmarks = []
    for i in range(21):
        x = row[f'x{i}']
        y = row[f'y{i}']
        z = row[f'z{i}']
        landmarks.append((x, y, z))
    
    features = []
    
    # MediaPipe hand landmark indices:
    # 0: WRIST
    # 1-4: THUMB (1=CMC, 2=MCP, 3=IP, 4=TIP)
    # 5-8: INDEX (5=MCP, 6=PIP, 7=DIP, 8=TIP)
    # 9-12: MIDDLE (9=MCP, 10=PIP, 11=DIP, 12=TIP)
    # 13-16: RING (13=MCP, 14=PIP, 15=DIP, 16=TIP)
    # 17-20: PINKY (17=MCP, 18=PIP, 19=DIP, 20=TIP)
    
    # 1. Distance features
    wrist = landmarks[0]
    
    # Distances from wrist to fingertips
    fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    for tip_idx in fingertips:
        features.append(calculate_distance(wrist, landmarks[tip_idx]))
    
    # Distances between fingertips
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            features.append(calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[j]]))
    
    # Distances from wrist to finger MCPs (base joints)
    mcps = [1, 5, 9, 13, 17]  # thumb CMC, index MCP, middle MCP, ring MCP, pinky MCP
    for mcp_idx in mcps:
        features.append(calculate_distance(wrist, landmarks[mcp_idx]))
    
    # 2. Finger length features
    finger_segments = [
        [1, 2, 3, 4],    # thumb
        [5, 6, 7, 8],    # index
        [9, 10, 11, 12], # middle
        [13, 14, 15, 16], # ring
        [17, 18, 19, 20] # pinky
    ]
    
    for finger in finger_segments:
        # Total finger length (sum of segment lengths)
        total_length = 0
        for i in range(len(finger)-1):
            total_length += calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]])
        features.append(total_length)
          # Individual segment lengths
        for i in range(len(finger)-1):
            features.append(calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]]))
    
    # 3. Angle features (finger curvature)
    for finger in finger_segments:
        # Calculate angles at each joint
        for i in range(1, len(finger)-1):
            angle = calculate_angle(landmarks[finger[i-1]], landmarks[finger[i]], landmarks[finger[i+1]])
            features.append(angle)
    
    # 4. Inter-finger angles (angles between adjacent fingers)
    # Angle between thumb and index
    features.append(calculate_angle(landmarks[4], landmarks[0], landmarks[8]))
    # Angle between index and middle
    features.append(calculate_angle(landmarks[8], landmarks[0], landmarks[12]))
    # Angle between middle and ring
    features.append(calculate_angle(landmarks[12], landmarks[0], landmarks[16]))
    # Angle between ring and pinky
    features.append(calculate_angle(landmarks[16], landmarks[0], landmarks[20]))
    
    # 5. Palm-related features
    # Palm center (approximate)
    palm_points = [0, 1, 5, 9, 13, 17]  # wrist and MCPs
    palm_center_x = np.mean([landmarks[i][0] for i in palm_points])
    palm_center_y = np.mean([landmarks[i][1] for i in palm_points])
    palm_center_z = np.mean([landmarks[i][2] for i in palm_points])
    palm_center = (palm_center_x, palm_center_y, palm_center_z)
    
    # Distances from palm center to fingertips
    for tip_idx in fingertips:
        features.append(calculate_distance(palm_center, landmarks[tip_idx]))
    
    # 6. Statistical features
    # Variance of fingertip distances from wrist
    tip_distances = [calculate_distance(wrist, landmarks[tip_idx]) for tip_idx in fingertips]
    features.append(np.var(tip_distances))
    features.append(np.mean(tip_distances))
    features.append(np.std(tip_distances))
    
    # 7. Hand orientation features
    # Vector from wrist to middle finger MCP
    wrist_to_middle = np.array([landmarks[9][0] - landmarks[0][0], 
                               landmarks[9][1] - landmarks[0][1], 
                               landmarks[9][2] - landmarks[0][2]])
    
    # Hand direction angles
    features.append(math.atan2(wrist_to_middle[1], wrist_to_middle[0]))  # yaw
    features.append(math.atan2(wrist_to_middle[2], 
                              math.sqrt(wrist_to_middle[0]**2 + wrist_to_middle[1]**2)))  # pitch
    
    # 8. Spread features
    # Maximum spread (distance between farthest fingertips)
    max_spread = 0
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            dist = calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[j]])
            max_spread = max(max_spread, dist)
    features.append(max_spread)
    
    # Thumb-pinky spread
    features.append(calculate_distance(landmarks[4], landmarks[20]))
    
    # 9. Height features (z-coordinate statistics)
    z_coords = [landmarks[i][2] for i in range(21)]
    features.append(np.mean(z_coords))
    features.append(np.std(z_coords))
    features.append(max(z_coords) - min(z_coords))  # z-range
    
    return features

def main():
    print("Loading hand gesture training data...")
    
    # Load the CSV file
    df = pd.read_csv('d:/HAND_GESTURE/Data/hand_gesture_training_data.csv')
    
    print(f"Loaded {len(df)} samples")
    print(f"Unique labels: {df['label'].unique()}")
    
    # Extract features for all samples
    print("Extracting features...")
    all_features = []
    labels = []
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Processing sample {idx}/{len(df)}")
        
        features = extract_features(row)
        all_features.append(features)
        labels.append(row['label'])
    
    # Convert to numpy array
    feature_matrix = np.array(all_features)
    
    print(f"Extracted {feature_matrix.shape[1]} features per sample")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    # Create feature names for better understanding
    feature_names = []
    
    # Distance features (5 + 10 + 5 = 20)
    fingertip_names = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
    for tip in fingertip_names:
        feature_names.append(f'wrist_to_{tip}_distance')
    
    for i in range(len(fingertip_names)):
        for j in range(i+1, len(fingertip_names)):
            feature_names.append(f'{fingertip_names[i]}_to_{fingertip_names[j]}_distance')
    
    mcp_names = ['thumb_cmc', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinky_mcp']
    for mcp in mcp_names:
        feature_names.append(f'wrist_to_{mcp}_distance')
    
    # Finger length features (5 total lengths + 15 individual segments = 20)
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for finger in finger_names:
        feature_names.append(f'{finger}_total_length')
    
    for finger in finger_names:
        if finger == 'thumb':
            segments = ['cmc_to_mcp', 'mcp_to_ip', 'ip_to_tip']
        else:
            segments = ['mcp_to_pip', 'pip_to_dip', 'dip_to_tip']
        for segment in segments:
            feature_names.append(f'{finger}_{segment}_length')
    
    # Angle features (3*5 = 15)
    for finger in finger_names:
        if finger == 'thumb':
            joints = ['mcp', 'ip']
        else:
            joints = ['pip', 'dip']
        for joint in joints:
            feature_names.append(f'{finger}_{joint}_angle')
    
    # Inter-finger angles (4)
    feature_names.extend(['thumb_index_angle', 'index_middle_angle', 'middle_ring_angle', 'ring_pinky_angle'])
    
    # Palm features (5)
    for tip in fingertip_names:
        feature_names.append(f'palm_center_to_{tip}_distance')
    
    # Statistical features (3)
    feature_names.extend(['fingertip_distance_variance', 'fingertip_distance_mean', 'fingertip_distance_std'])
    
    # Orientation features (2)
    feature_names.extend(['hand_yaw', 'hand_pitch'])
    
    # Spread features (2)
    feature_names.extend(['max_fingertip_spread', 'thumb_pinky_spread'])
    
    # Height features (3)
    feature_names.extend(['z_coord_mean', 'z_coord_std', 'z_coord_range'])
    
    # Create DataFrame with features
    feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
    feature_df['label'] = labels
    
    # Save to CSV
    output_path = 'd:/HAND_GESTURE/Data.csv'
    feature_df.to_csv(output_path, index=False)
    
    print(f"\nFeature extraction completed!")
    print(f"Saved {len(feature_df)} samples with {len(feature_names)} features to {output_path}")
    print(f"\nFeature summary:")
    print(f"- Distance features: 20")
    print(f"- Finger length features: 20") 
    print(f"- Angle features: 15")
    print(f"- Inter-finger angles: 4")
    print(f"- Palm-related features: 5")
    print(f"- Statistical features: 3")
    print(f"- Orientation features: 2")
    print(f"- Spread features: 2")
    print(f"- Height features: 3")
    print(f"- Total features: {len(feature_names)}")
    
    # Display sample statistics
    print(f"\nDataset statistics:")
    print(feature_df['label'].value_counts())
    
    print(f"\nFirst few feature values:")
    print(feature_df.head())

if __name__ == "__main__":
    main()
