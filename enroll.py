# enroll.py — enroll YOUR wrist as the authorized user

import cv2
import numpy as np
import pickle
import os
from preprocess import preprocess_vein

ENROLL_COUNT = 20       # uses first 20 images for enrollment
                        # remaining images in data/me/ are used for testing in evaluate.py
ME_FOLDER    = "data/me"


def enroll_user(user_name, image_folder, max_images=ENROLL_COUNT):

    print(f"\n{'='*52}")
    print(f"  Enrolling   : {user_name}")
    print(f"  Folder      : {image_folder}")
    print(f"  Using first : {max_images} images")
    print(f"{'='*52}\n")

    if not os.path.exists(image_folder):
        print(f"  ERROR: Folder not found → {image_folder}")
        print(f"  Run capture.py first to collect images.")
        return

    files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not files:
        print("  ERROR: No images found in folder.")
        print("  Run capture.py first to collect images.")
        return

    if len(files) < max_images:
        print(f"  WARNING: Only {len(files)} images found, need {max_images}.")
        print(f"  Enrolling with all {len(files)} available images.\n")
        max_images = len(files)

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
            print(f"  [{i+1:>2}] SKIP  {fname}  — too few keypoints ({len(kp) if kp else 0})")

    if not all_descriptors:
        print("\n  FAILED: No valid images enrolled.")
        print("  Check image quality — make sure IR LEDs are on and wrist is in frame.")
        return

    template  = np.vstack(all_descriptors)
    save_path = f"data/templates/{user_name}_template.pkl"
    os.makedirs("data/templates", exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(template, f)

    print(f"\n{'='*52}")
    print(f"  Enrolled successfully!")
    print(f"  User              : {user_name}")
    print(f"  Template saved to : {save_path}")
    print(f"  Total descriptors : {len(template)}")
    print(f"{'='*52}\n")
    print(f"  Next step: run evaluate.py to calculate the unlock threshold.\n")


if __name__ == "__main__":
    enroll_user("me", ME_FOLDER, max_images=ENROLL_COUNT)
