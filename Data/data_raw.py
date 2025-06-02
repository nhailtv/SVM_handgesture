
import os
import cv2
import csv
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tqdm import tqdm

# Directories
input_folder = r'D:\HAND_GESTURE\Data\hagrid'  

output_img_folder = r'Data\hand_gesture_picture'
output_csv = 'Data\hand_gesture_training_data.csv'

# Create folders if not exist
os.makedirs(output_img_folder, exist_ok=True)

# Initialize hand detector
try:
    detector = HandDetector(detectionCon=0.8, maxHands=1)
except Exception as e:
    print("HandDetector module not found or failed to initialize.", e)
    detector = None


# Function to process images with options
def are_landmarks_close(lm1, lm2, threshold=15):
    if len(lm1) != len(lm2):
        return False
    # Calculate mean Euclidean distance between corresponding points
    dists = [np.linalg.norm(np.array(lm1[i]) - np.array(lm2[i])) for i in range(len(lm1))]
    return np.mean(dists) < threshold

def process_hand_gesture_images(max_images_per_class=None, use_gpu=False):
    if not detector:
        print("HandDetector not available. Skipping image processing.")
        return

    # Đếm tổng số ảnh cần xử lý để hiển thị progress
    total_images = 0
    gesture_folders = []
    
    print("📁 Đang quét thư mục để đếm số ảnh...")
    for gesture_name in os.listdir(input_folder):
        gesture_path = os.path.join(input_folder, gesture_name)
        if os.path.isdir(gesture_path):
            image_files = [f for f in os.listdir(gesture_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_images_per_class is not None:
                images_to_process = min(len(image_files), max_images_per_class)
            else:
                images_to_process = len(image_files)
            
            total_images += images_to_process
            gesture_folders.append((gesture_name, gesture_path, image_files[:images_to_process]))
            print(f"  📂 {gesture_name}: {images_to_process} ảnh")
    
    print(f"\n🎯 Tổng cộng sẽ xử lý: {total_images} ảnh")
    print("🚀 Bắt đầu xử lý...\n")
    
    # Progress bar cho toàn bộ quá trình

    # Prepare output CSV
    with open(output_csv, 'w', newline='') as out_csv:
        writer = csv.writer(out_csv)
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        header.append('label')
        writer.writerow(header)

        with tqdm(total=total_images, desc="🖼️ Xử lý ảnh", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}') as pbar:
            total_processed = 0
            total_successful = 0
            for gesture_name, gesture_path, image_files in gesture_folders:
                images_processed = 0
                gesture_successful = 0
                for filename in image_files:
                    img_path = os.path.join(gesture_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        # First detection
                        hands1, img_with_hand1 = detector.findHands(img.copy())
                        time.sleep(0.1)  # Give more time for detection
                        # Second detection
                        hands2, img_with_hand2 = detector.findHands(img.copy())

                        if hands1 and hands2:
                            lm_list1 = hands1[0]['lmList']
                            lm_list2 = hands2[0]['lmList']
                            if are_landmarks_close(lm_list1, lm_list2, threshold=15):
                                # Draw landmarks on the original image
                                for lm in lm_list1:
                                    cv2.circle(img_with_hand1, (int(lm[0]), int(lm[1])), 5, (0, 255, 0), -1)

                                # Save the image with landmarks
                                img_output_name = f"{gesture_name}_{images_processed+1}.jpg"
                                out_img_path = os.path.join(output_img_folder, img_output_name)
                                cv2.imwrite(out_img_path, img_with_hand1)

                                # Write landmarks directly to the merged CSV
                                row_data = []
                                for point in lm_list1:
                                    row_data.extend([point[0], point[1], point[2]])
                                row_data.append(gesture_name)
                                writer.writerow(row_data)

                                gesture_successful += 1
                                total_successful += 1
                        images_processed += 1
                        total_processed += 1
                    except Exception as e:
                        images_processed += 1
                        total_processed += 1

                    # Cập nhật progress bar với thông tin chi tiết
                    pbar.set_postfix({
                        'Gesture': gesture_name,
                        'Success': f"{total_successful}/{total_processed}",
                        'Current': f"{gesture_successful}/{images_processed}"
                    })
                    pbar.update(1)

    print(f"\n✅ Hoàn thành! Đã xử lý thành công {total_successful}/{total_processed} ảnh")
    print(f"📸 Raw image and landmark extraction completed. Output CSV: {output_csv}")


# Main execution
if __name__ == "__main__":
    print("🚀 Bắt đầu xử lý ảnh cử chỉ tay...")
process_hand_gesture_images(max_images_per_class=1000)

