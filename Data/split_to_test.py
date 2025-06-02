import os
import random
import shutil

src_root = r'd:\HAND_GESTURE\Data\hagrid'
dst_root = r'd:\HAND_GESTURE\Data\hagrid_test'

for class_name in os.listdir(src_root):
    src_dir = os.path.join(src_root, class_name)
    dst_dir = os.path.join(dst_root, class_name)
    if not os.path.isdir(src_dir):
        continue
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    n_test = int(len(images) * 0.3)
    test_images = images[:n_test]
    for img in test_images:
        src_img = os.path.join(src_dir, img)
        dst_img = os.path.join(dst_dir, img)
        shutil.move(src_img, dst_img)
print('Done splitting images to test folders.')
