# capture.py — collect wrist images using NoIR camera on RPi4

from picamera2 import Picamera2
import time
import os
import sys


def collect_images(save_folder, count=30):

    os.makedirs(save_folder, exist_ok=True)

    # Check how many images already exist so we don't overwrite
    existing = sorted([
        f for f in os.listdir(save_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    start_index = len(existing) + 1

    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": (1280, 960)}   # good resolution for vein detail
    )
    cam.configure(config)
    cam.start()
    time.sleep(2)  # warm up camera

    print(f"\n{'='*52}")
    print(f"  BioRhythm Lock — Image Capture")
    print(f"  Saving to   : {save_folder}")
    print(f"  Target count: {count} images")
    print(f"  Starting at : wrist_{start_index:03d}.jpg")
    print(f"{'='*52}")
    print(f"\n  Place your wrist under the IR LEDs.")
    print(f"  Keep your wrist STILL when capturing.")
    print(f"  Press ENTER to capture. Type 'q' + ENTER to quit.\n")

    captured = 0
    i = start_index

    while captured < count:
        inp = input(f"  [{captured+1}/{count}]  Press ENTER to capture (or q to quit): ")

        if inp.strip().lower() == 'q':
            print("\n  Quit early.")
            break

        path = os.path.join(save_folder, f"wrist_{i:03d}.jpg")
        cam.capture_file(path)
        print(f"    ✓ Saved → {path}\n")
        captured += 1
        i += 1

    cam.stop()

    print(f"\n{'='*52}")
    print(f"  Done! {captured} images saved to: {save_folder}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    # Usage:
    #   python3 capture.py          → saves to data/me   (default)
    #   python3 capture.py data/me
    #   python3 capture.py data/friend

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/me"
    collect_images(folder, count=30)