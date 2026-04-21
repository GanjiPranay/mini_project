# 🔐 BioRhythm Lock — Wrist Vein Biometric Authentication System

> A near-infrared (NIR) based biometric authentication system that captures, processes, and matches wrist vein patterns for identity verification — built using a Raspberry Pi, an IR-sensitive camera, and a custom Python image processing pipeline.

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [How It Works — Big Picture](#2-how-it-works--big-picture)
3. [Hardware Architecture](#3-hardware-architecture)
   - 3.1 [Why Near-Infrared (NIR)?](#31-why-near-infrared-nir)
   - 3.2 [Components](#32-components)
   - 3.3 [LED Positioning — The Critical Detail](#33-led-positioning--the-critical-detail)
   - 3.4 [Camera Filter](#34-camera-filter)
   - 3.5 [Current Hardware Status & Planned Fixes](#35-current-hardware-status--planned-fixes)
4. [Software Pipeline](#4-software-pipeline)
   - 4.1 [Image Preprocessing](#41-image-preprocessing)
   - 4.2 [Feature Extraction](#42-feature-extraction)
   - 4.3 [Matching & Authentication](#43-matching--authentication)
5. [Evaluation Methodology](#5-evaluation-methodology)
   - 5.1 [FAR & FRR](#51-far--frr)
   - 5.2 [Dataset Requirements](#52-dataset-requirements)
6. [Known Limitations & Issues](#6-known-limitations--issues)
7. [Project Structure](#7-project-structure)
8. [Setup & Installation](#8-setup--installation)
   - 8.1 [Raspberry Pi OS Setup](#81-raspberry-pi-os-setup)
   - 8.2 [Camera Setup](#82-camera-setup)
   - 8.3 [Python Dependencies](#83-python-dependencies)
9. [Running the System](#9-running-the-system)
10. [Roadmap](#10-roadmap)
11. [Team](#11-team)

---

## 1. Project Overview

**BioRhythm Lock** is a hardware-software biometric authentication system that identifies individuals using the unique vein patterns in their wrists.

Unlike fingerprint or face recognition, vein-based biometrics are:
- **Harder to spoof** — veins are internal and not exposed on the surface
- **More hygienic** — contactless scanning
- **Unique per person** — even identical twins have different vein patterns

The system illuminates the wrist with **850nm near-infrared light**, captures the resulting vein image with an IR-sensitive camera, processes it through a multi-stage software pipeline, and then matches extracted features against a stored template to grant or deny access.

**Current Status:**
| Component | Status |
|---|---|
| Software Pipeline | ✅ Complete |
| Hardware (Camera + LEDs) | 🔧 Pending final assembly with correct components |
| Evaluation (FAR/FRR) | ⚠️ Partially implemented — needs dataset expansion |

---

## 2. How It Works — Big Picture

```
        ┌─────────────────────────────────────────────────────┐
        │                  ENROLLMENT PHASE                   │
        │  Place wrist → Capture 45–50 NIR images → Process  │
        │  → Extract features → Store as reference template   │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │               AUTHENTICATION PHASE                  │
        │  Place wrist → Capture NIR image → Process         │
        │  → Extract features → Match vs stored template     │
        │  → Score above threshold? → GRANT / DENY access    │
        └─────────────────────────────────────────────────────┘
```

The pipeline between "capture" and "features" involves five key stages:

```
Raw NIR Image
     │
     ▼
[1] CLAHE Enhancement        ← Boost local contrast in vein regions
     │
     ▼
[2] Median Blur              ← Remove noise while preserving vein edges
     │
     ▼
[3] Otsu Thresholding        ← Convert to binary (vein vs background)
     │
     ▼
[4] Skeletonization          ← Thin vein regions to single-pixel paths
     │
     ▼
[5] ORB Feature Matching     ← Extract & match keypoints for identity check
```

---

## 3. Hardware Architecture

### 3.1 Why Near-Infrared (NIR)?

Hemoglobin in blood **absorbs near-infrared light** more strongly than the surrounding tissue. When 850nm IR LEDs illuminate the wrist from the side, veins appear as **dark lines** against a lighter background in the captured image. This contrast is invisible to the human eye but clearly detectable by an IR-sensitive camera.

```
        Side view of wrist:

        [LED] ──── NIR light ────▶ [Wrist]
                                      │
                          Veins absorb more IR
                          Skin reflects more IR
                                      │
                                      ▼
                         Camera captures contrast map
```

### 3.2 Components

| Component | Details |
|---|---|
| **Compute Board** | Raspberry Pi 4 or 5 (college-owned) |
| **Camera** | Raspberry Pi NoIR Camera Module V2 (8MP, Sony IMX219 sensor) |
| **IR LEDs** | 6× 850nm IR LEDs |
| **Bandpass Filter** | 8×8mm 850nm bandpass filter (glass) |
| **CSI Cable** | 15-to-22 pin adapter (required for RPi 5 only; RPi 4 is direct plug-in) |

**Why the NoIR Camera Module V2?**
The "NoIR" (No Infrared filter) variant has the IR cut filter physically removed from the sensor. A standard camera blocks near-infrared light — the NoIR module lets it through, making it ideal for vein imaging without any hardware modification.

### 3.3 LED Positioning — The Critical Detail

LED placement is the **single most important factor** in vein image quality. Getting this wrong results in inconsistent, unusable images.

**❌ Wrong — LEDs pointing downward from above:**
```
        [LED] [LED] [LED]
          ↓     ↓     ↓
        ═══════════════    ← Wrist (top-lit, poor contrast, flat shadows)
```
This creates flat, uneven illumination with poor vein-to-skin contrast. Veins are hard to distinguish.

**✅ Correct — LEDs side-mounted, angled 20–30° inward:**
```
         [LED]             [LED]
           ↘  20–30°   ↙
           ════════════    ← Wrist (side-lit, deep contrast, veins visible)
```
Side-mounted LEDs send NIR light **through** the wrist tissue at an angle. Veins, which absorb more IR, show up as distinct dark lines. This is the standard approach in research-grade vein scanners.

> ⚠️ The current prototype uses downward-facing LEDs, which has been identified as the root cause of inconsistent vein images. The fix (side-mounting at 20–30°) is pending hardware reassembly.

### 3.4 Camera Filter

An **850nm bandpass filter** is placed in front of the camera lens to:
- Block all visible light from entering the sensor
- Allow only 850nm IR light (reflected from the wrist) to pass through
- Eliminate ambient light interference, which would wash out vein contrast

**Filter size matters:** Standard DSLR-sized filters are incompatible with the small NoIR camera lens. The correct choice is a small **8×8mm glass bandpass filter** (available on Amazon.in for ₹150–300).

### 3.5 Current Hardware Status & Planned Fixes

| Item | Current State | Fix |
|---|---|---|
| Camera | Modified webcam (IR cut filter manually removed) | Replace with RPi NoIR Camera Module V2 |
| IR Filter | Exposed negative film as makeshift NIR filter | Replace with 8×8mm 850nm bandpass glass filter |
| LED Position | Downward-facing from above | Remount side-facing, angled 20–30° inward |
| CSI Cable | — | 15-to-22 pin adapter needed for RPi 5 |

---

## 4. Software Pipeline

All software is written in Python. The pipeline runs on the Raspberry Pi and consists of five sequential stages.

### 4.1 Image Preprocessing

Raw NIR wrist images have low contrast and noise that must be corrected before feature extraction.

#### Stage 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

Standard histogram equalization boosts contrast globally, which can over-amplify noise. CLAHE divides the image into small tiles and equalizes each region independently with a contrast cap (clip limit). This brings out local vein structure without blowing out brighter regions.

```
Input: Low-contrast grayscale NIR image
Output: Locally enhanced image with visible vein-skin contrast
```

#### Stage 2 — Median Blur

A median filter replaces each pixel with the **median value** of its neighborhood. Unlike a Gaussian blur, it removes salt-and-pepper noise without smearing edges — preserving the sharp boundaries of vein walls.

```
Input: CLAHE-enhanced image
Output: Smooth image with noise removed, edges intact
```

#### Stage 3 — Otsu Thresholding

Otsu's method automatically finds the optimal threshold value that separates the image into two classes: **veins (dark)** and **background (light)**. The result is a binary image — pixels are either 0 (vein) or 255 (background).

```
Input: Blurred grayscale image
Output: Binary image — pure black veins on white background
```

### 4.2 Feature Extraction

#### Stage 4 — Skeletonization

The binary vein map is thinned to **single-pixel-wide paths** using morphological skeletonization. This removes the width variation in veins (which can change based on blood pressure, temperature, etc.) and preserves only the **topological structure** — branching points, endpoints, and paths — which are stable across scans.

```
Input: Binary vein mask
Output: Skeleton image — 1-pixel-wide vein paths
```

#### Stage 5 — ORB Feature Matching

ORB (Oriented FAST and Rotated BRIEF) is a classical computer vision feature detector and descriptor. It:
1. Detects **keypoints** (interesting structural points) in the skeleton image
2. Computes a **descriptor** (binary string representing local structure around each keypoint)
3. At match time, descriptors from the query image are compared to stored descriptors using **Hamming distance**
4. The number of **good matches** (below a distance threshold) determines the similarity score

```
Input: Skeleton image
Output: Set of keypoints + binary descriptors
Match score: Number of good descriptor matches between query and template
```

### 4.3 Matching & Authentication

At authentication time:
1. A new wrist image is captured and processed through all five stages
2. Its ORB descriptors are extracted
3. They are matched against the stored enrollment template descriptors
4. If the match score exceeds a defined threshold → **Access Granted**
5. If it falls below the threshold → **Access Denied**

---

## 5. Evaluation Methodology

### 5.1 FAR & FRR

The system is evaluated using two standard biometric metrics:

| Metric | Full Name | What it Measures |
|---|---|---|
| **FAR** | False Acceptance Rate | % of impostors incorrectly accepted |
| **FRR** | False Rejection Rate | % of genuine users incorrectly rejected |

A lower FAR = better security. A lower FRR = better convenience. The threshold can be tuned to trade off between them.

```
            FAR                       FRR
             │                         │
   Impostors │ incorrectly       Genuine│ incorrectly
   accepted  │ granted access     users │ denied access
             │                         │
         BAD for SECURITY          BAD for USABILITY
```

### 5.2 Dataset Requirements

| Category | Minimum Recommended | Current State |
|---|---|---|
| Self (genuine) images — enrollment | 45–50 | To be captured |
| Self (genuine) images — test set | 10–15 | To be captured |
| Impostor images (other people) | 20–30 minimum | Only 5–6 (insufficient) |

**Evaluation strategy:**
- Train (enroll) on ~45–50 images of the authorized user
- Test on remaining self images → compute **FRR**
- Test on impostor (friend) images → compute **FAR**
- The `evaluate.py` script handles this, but currently has a bug (see Known Limitations)

---

## 6. Known Limitations & Issues

### 🐛 Bug: `evaluate.py` Incorrect Folder Path for Impostors

In the current `evaluate.py`, the impostor scoring path points to the **wrong folder**. This means the FAR calculation is being run on incorrect data, producing unreliable results.

**Fix:** Update the folder path variable in `evaluate.py` to correctly point to the directory containing impostor (friend) images.

---

### ⚠️ Insufficient Impostor Dataset

Only 5–6 friend images are currently available as impostor samples. This is far below the recommended minimum of 20–30. With so few samples, the FAR metric is statistically unreliable and likely to be misleading.

**Fix:** Collect wrist images from at least 4–5 different people (20–30 images total) for a credible FAR evaluation.

---

### ⚠️ ORB is Suboptimal for Vein Skeleton Patterns

ORB was designed for textured, gradient-rich natural images (faces, objects, scenes). Vein skeletons are **sparse, thin-line structures** with very few traditional keypoints. ORB often fails to find enough stable, repeatable keypoints on a skeleton image, leading to:
- Low match scores even for genuine users (high FRR)
- Inconsistent results across scans of the same person

**Better alternatives to consider:**
| Algorithm | Why Better for Veins |
|---|---|
| **Minutiae-based matching** | Designed specifically for ridge/vein branching structures |
| **SIFT** | More robust keypoints on low-texture images (but slower) |
| **Template correlation** | Direct pixel-to-pixel comparison of skeleton images |
| **Deep learning (CNN embedding)** | Learns vein-specific features end-to-end |

For a college project, **template correlation or SIFT** would be a practical improvement over ORB.

---

### ⚠️ Hardware: LED Positioning

As described in Section 3.3, the current downward-facing LED configuration is the primary cause of inconsistent vein image quality. This is a hardware issue — no amount of software tuning can fully compensate for poor illumination.

---

### ℹ️ `picamera` vs `picamera2`

Older Raspberry Pi tutorials use the `picamera` library, which is **no longer supported** on current Raspberry Pi OS. The correct library is `picamera2`. Any code using `import picamera` will fail on a fresh RPi OS installation.

---

## 7. Project Structure

```
biorhythm-lock/
│
├── capture/
│   └── capture.py          # Captures wrist images using picamera2
│
├── preprocessing/
│   ├── clahe.py            # CLAHE contrast enhancement
│   ├── blur.py             # Median blur noise removal
│   └── threshold.py        # Otsu thresholding → binary image
│
├── features/
│   ├── skeletonize.py      # Morphological skeletonization
│   └── orb.py              # ORB keypoint extraction & descriptor computation
│
├── matching/
│   └── matcher.py          # Descriptor matching & score computation
│
├── enrollment/
│   └── enroll.py           # Full pipeline: capture → process → save template
│
├── evaluation/
│   └── evaluate.py         # FAR/FRR computation (⚠️ has folder path bug)
│
├── templates/
│   └── user_template.npy   # Stored ORB descriptors for enrolled user
│
├── data/
│   ├── self/               # Genuine user images (45–50 enrollment + test)
│   └── impostors/          # Friend/impostor images (needs 20–30 minimum)
│
├── requirements.txt
└── README.md
```

> Note: Folder structure above is the recommended layout. Actual file organization in your repository may vary.

---

## 8. Setup & Installation

### 8.1 Raspberry Pi OS Setup

If you are new to Raspberry Pi, the recommended setup flow is:

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Flash **Raspberry Pi OS (64-bit, Bookworm)** onto a microSD card (minimum 16GB)
3. During flashing, configure: hostname, username/password, Wi-Fi credentials, and enable SSH
4. Boot the Pi and connect via SSH or with a monitor/keyboard

**Recommended learning resources for RPi beginners:**
- **NetworkChuck** — Getting started with Raspberry Pi (YouTube)
- **Jeff Geerling** — In-depth RPi tutorials and benchmarks (YouTube)
- **ExplainingComputers** — Hardware-focused RPi walkthroughs (YouTube)

### 8.2 Camera Setup

**For Raspberry Pi 4:**
The NoIR Camera Module V2 plugs directly into the CSI camera port (15-pin ribbon cable, no adapter needed).

**For Raspberry Pi 5:**
The RPi 5 uses a **22-pin CSI connector** while the Camera Module V2 has a **15-pin ribbon cable**. You need a **15-to-22 pin CSI adapter cable** (~₹100–150).

**Enable the camera:**
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
sudo reboot
```

**Test the camera:**
```bash
libcamera-hello --timeout 5000
```
If you see a live preview for 5 seconds, the camera is working correctly.

**Install picamera2:**
```bash
sudo apt update
sudo apt install -y python3-picamera2
```

> ⚠️ Do NOT install or use the old `picamera` library. It is deprecated and incompatible with current Raspberry Pi OS. Always use `picamera2`.

### 8.3 Python Dependencies

```bash
pip install opencv-python numpy scikit-image
```

| Library | Purpose |
|---|---|
| `opencv-python` | CLAHE, blur, thresholding, ORB |
| `numpy` | Array operations, template storage |
| `scikit-image` | Morphological skeletonization |
| `picamera2` | Raspberry Pi camera interface |

Install all at once:
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
opencv-python
numpy
scikit-image
picamera2
```

---

## 9. Running the System

### Step 1 — Enroll a User (Capture Training Images)

```bash
python enrollment/enroll.py --user self --count 50
```

Place your wrist under the camera in a consistent position. The script will capture 50 images, process each through the full pipeline, and save the resulting feature template.

### Step 2 — Capture Impostor Images

Repeat the capture process for at least 4–5 different people:
```bash
python capture/capture.py --output data/impostors/ --count 5
```

Do this for each impostor person (aim for 20–30 total impostor images).

### Step 3 — Run Evaluation

```bash
python evaluation/evaluate.py
```

This will output:
- **FRR** — False Rejection Rate (how often your own wrist is rejected)
- **FAR** — False Acceptance Rate (how often an impostor's wrist is accepted)

> ⚠️ Fix the folder path bug in `evaluate.py` before running this step.

### Step 4 — Authenticate

```bash
python matching/matcher.py
```

Place your wrist in front of the camera. The system will capture an image, process it, match it against the stored template, and print `ACCESS GRANTED` or `ACCESS DENIED`.

---

## 10. Roadmap

| Priority | Task | Status |
|---|---|---|
| 🔴 High | Reassemble hardware with correct LED side-mounting | Pending |
| 🔴 High | Acquire RPi NoIR Camera Module V2 | Pending |
| 🔴 High | Acquire 8×8mm 850nm bandpass filter | Pending |
| 🔴 High | Fix `evaluate.py` impostor folder path bug | Pending |
| 🟡 Medium | Expand impostor dataset to 20–30 images | Pending |
| 🟡 Medium | Implement full FAR/FRR evaluation and plot EER curve | Pending |
| 🟢 Low | Evaluate SIFT or template correlation as ORB replacement | Future |
| 🟢 Low | Add threshold tuning UI | Future |

---

## 11. Team

| Name | Role |
|---|---|
| Pranay | Hardware design, system integration, documentation |
| Teammate | Software pipeline (preprocessing, feature extraction, matching, evaluation) |

**Hardware:** Raspberry Pi 4/5 (college-owned)
**Camera source:** [Robu.in — RPi NoIR Camera Module V2](https://robu.in) (~₹1,585)
**Filter source:** Amazon.in — 8×8mm 850nm bandpass filter (~₹150–300)

---

## License

This project was developed as part of a college mini-project. All code is for educational purposes.
