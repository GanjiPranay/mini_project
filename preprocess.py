# preprocess.py

import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


def preprocess_vein(image_path, show_steps=False):

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load → {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE — boost local contrast so faint veins appear
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Median blur — remove IR sensor grain noise
    blurred = cv2.medianBlur(clahe_img, 5)

    # Otsu binarization — auto threshold, veins=white skin=black
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological closing — seal tiny gaps in vein lines
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Skeletonize — reduce veins to 1-pixel-wide lines
    skeleton     = skeletonize(closed // 255)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    if show_steps:
        steps  = [gray, clahe_img, blurred, binary, closed, skeleton_img]
        titles = ['1.Gray','2.CLAHE','3.Blur','4.Otsu','5.Close','6.Skeleton']
        fig, axes = plt.subplots(1, 6, figsize=(22, 4))
        for ax, title, image in zip(axes, titles, steps):
            ax.imshow(image, cmap='gray')
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        plt.suptitle(image_path, fontsize=10)
        plt.tight_layout()
        plt.show()

    return skeleton_img