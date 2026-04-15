# match.py
import cv2
import numpy as np
import pickle
import sys
import os
from preprocess import preprocess_vein

THRESHOLD_FILE = "data/templates/threshold.txt"

def get_threshold():
    """Read threshold calculated by evaluate.py"""
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            t = int(f.read().strip())
            print(f"  Threshold loaded : {t} (from evaluate.py)")
            return t
    except FileNotFoundError:
        print("  WARNING: threshold.txt not found!")
        print("  Run evaluate.py first to calculate the correct threshold.")
        print("  Using fallback threshold = 100")
        return 100

def match_wrist(image_path):
    print(f"\n{'='*52}")
    print(f"  BioRhythm Lock — Scanning...")
    print(f"  Image : {image_path}")
    print(f"{'='*52}")

    # Load threshold automatically
    THRESHOLD = get_threshold()

    # Preprocess
    skeleton = preprocess_vein(image_path, show_steps=False)
    if skeleton is None:
        print("  ERROR: Could not process image.")
        return

    # Extract features
    orb      = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(skeleton, None)
    if desc is None or len(kp) < 10:
        print(f"\n  Score     : 0")
        print(f"  Threshold : {THRESHOLD}")
        print(f"\n{'='*52}")
        print("  ACCESS DENIED — image too unclear to read")
        print(f"{'='*52}\n")
        return

    # Load template
    try:
        with open("data/templates/friend_template.pkl", 'rb') as f:
            template = pickle.load(f)
    except FileNotFoundError:
        print("  ERROR: No template found. Run enroll.py first!")
        return

    # Match
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc, template)
    good    = [m for m in matches if m.distance < 60]
    score   = len(good)
    margin  = score - THRESHOLD

    print(f"\n  Features detected : {len(kp)} keypoints")
    print(f"  Match score       : {score}")
    print(f"  Threshold         : {THRESHOLD}")
    print(f"  Margin            : {margin:+d}  "
          f"({'above' if margin >= 0 else 'below'} threshold)")
    print(f"\n{'='*52}")

    if score >= THRESHOLD:
        print("  ✅  ACCESS GRANTED 🔓")
        print("      Recognized as : FRIEND")
    else:
        print("  ❌  ACCESS DENIED  🔒")
        print("      Wrist not recognized.")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n  Usage:")
        print("  python match.py <image_path>")
        print("\n  Examples:")
        print("  python match.py data/friend/wrist_20.jpg")
        print("  python match.py data/me/wrist_01.jpg")
    else:
        match_wrist(sys.argv[1])