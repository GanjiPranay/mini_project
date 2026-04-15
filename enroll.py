# enroll.py  —  enrolls FRIEND ONLY
# -*- coding: utf-8 -*-

import cv2 
import numpy as np
import pickle
import os
from preprocess import preprocess_vein

ENROLL_COUNT = 15   # number of images used for enrollment


def enroll_user(user_name, image_folder, max_images=ENROLL_COUNT):

    print(f"\n{'='*52}")
    print(f"  Enrolling : {user_name}")
    print(f"  Folder    : {image_folder}")
    print(f"{'='*52}")

    if not os.path.exists(image_folder):
        print(f"  ERROR: Folder not found.")
        return

    files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not files:
        print("  ERROR: No images found in folder.")
        return

    print(f"  Using first {max_images} of {len(files)} images.\n")

    orb             = cv2.ORB_create(nfeatures=500)
    all_descriptors = []

    for i, fname in enumerate(files[:max_images]):
        path     = os.path.join(image_folder, fname)
        skeleton = preprocess_vein(path, show_steps=False)

        if skeleton is None:
            print(f"  [{i+1:>2}] SKIP  {fname}  — could not process")
            continue

        kp, desc = orb.detectAndCompute(skeleton, None)

        if desc is not None and len(kp) >= 10:
            all_descriptors.append(desc)
            print(f"  [{i+1:>2}] OK    {fname}  — {len(kp)} keypoints")
        else:
            print(f"  [{i+1:>2}] SKIP  {fname}  — too few keypoints")

    if not all_descriptors:
        print("\n  FAILED: No valid images. Check image quality.")
        return

    template  = np.vstack(all_descriptors)
    save_path = f"data/templates/{user_name}_template.pkl"
    os.makedirs("data/templates", exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(template, f)

    print(f"\n  Enrolled  : {user_name}")
    print(f"  Saved to  : {save_path}")
    print(f"  Total descriptors : {len(template)}")


if __name__ == "__main__":
    enroll_user("me", r"C:\Users\PRANAY\OneDrive\Desktop\Mini_project_1\Me", max_images=ENROLL_COUNT)