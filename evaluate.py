# evaluate.py — test accuracy and calculate unlock threshold

import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from preprocess import preprocess_vein

ENROLL_COUNT = 20       # must match enroll.py — these images are SKIPPED in testing
ME_FOLDER    = "data/me"
FRIEND_FOLDER = "data/friend"   # can be empty for now — handled gracefully


def get_score(image_path, template):
    """Score a single image against the enrolled template."""
    orb      = cv2.ORB_create(nfeatures=500)
    skeleton = preprocess_vein(image_path, show_steps=False)
    if skeleton is None:
        return 0
    _, desc = orb.detectAndCompute(skeleton, None)
    if desc is None:
        return 0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc, template)
    good    = [m for m in matches if m.distance < 60]
    return len(good)


def get_test_files(folder, skip=ENROLL_COUNT):
    """Return files from folder, skipping first `skip` (those were used for enrollment)."""
    if not os.path.exists(folder):
        return []
    all_f = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    return all_f[skip:]


def get_all_files(folder):
    """Return all image files from a folder."""
    if not os.path.exists(folder):
        return []
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


if __name__ == "__main__":

    # ── Load template ──────────────────────────────────────────────────────────
    try:
        with open("data/templates/me_template.pkl", 'rb') as f:
            template = pickle.load(f)
        print("\n  Template loaded successfully.")
    except FileNotFoundError:
        print("  ERROR: me_template.pkl not found.")
        print("  Run enroll.py first!")
        exit()

    # ── Get test files ─────────────────────────────────────────────────────────
    # My test images = images from data/me AFTER the first ENROLL_COUNT (which were used to train)
    me_test_files    = get_test_files(ME_FOLDER, skip=ENROLL_COUNT)

    # Friend images = ALL images from data/friend (none were used for training)
    friend_test_files = get_all_files(FRIEND_FOLDER)

    print(f"\n  My test images     : {len(me_test_files)}  (from data/me, skipping first {ENROLL_COUNT})")
    print(f"  Friend images      : {len(friend_test_files)}  (from data/friend)")

    if not me_test_files:
        print("\n  ERROR: No test images found!")
        print(f"  You need more than {ENROLL_COUNT} images in data/me/")
        print(f"  Currently only have images used for enrollment.")
        exit()

    # ── Score my images ────────────────────────────────────────────────────────
    me_scores = []
    print("\n  Scoring MY images (should score HIGH)...")
    for f in me_test_files:
        path  = os.path.join(ME_FOLDER, f)       # ← FIXED: correct folder
        score = get_score(path, template)
        me_scores.append(score)
        print(f"    {f:<30} →  {score:>4}  ✅ (authorized)")

    # ── Score friend images (impostor/stranger) ────────────────────────────────
    friend_scores = []
    if friend_test_files:
        print("\n  Scoring FRIEND images (should score LOW)...")
        for f in friend_test_files:
            path  = os.path.join(FRIEND_FOLDER, f)   # ← FIXED: correct folder
            score = get_score(path, template)
            friend_scores.append(score)
            print(f"    {f:<30} →  {score:>4}  ❌ (impostor)")
    else:
        print("\n  NOTE: No friend images found in data/friend/")
        print("  Threshold will be estimated from your own scores only.")
        print("  Add friend images later and re-run evaluate.py for accurate threshold.\n")

    # ── Threshold Calculation ──────────────────────────────────────────────────
    min_me       = min(me_scores) if me_scores else 0
    max_me       = max(me_scores) if me_scores else 0
    avg_me       = np.mean(me_scores) if me_scores else 0

    print(f"\n{'='*52}")
    print(f"  MY scores   →  min={min_me}  max={max_me}  avg={avg_me:.1f}")

    if friend_scores:
        max_friend   = max(friend_scores)
        min_friend   = min(friend_scores)
        avg_friend   = np.mean(friend_scores)
        gap          = min_me - max_friend

        print(f"  Friend scores →  min={min_friend}  max={max_friend}  avg={avg_friend:.1f}")
        print(f"  Gap (my min - friend max) = {gap}")

        if gap > 0:
            threshold = int(max_friend + gap * 0.5)
            print(f"\n  GAP EXISTS — system can tell you apart from friend!")
            print(f"  Threshold = friend max + (gap × 0.5) = {threshold}")
        else:
            threshold = int(min_me * 0.80)
            print(f"\n  WARNING: Scores overlap by {abs(gap)} points.")
            print(f"  Using fallback threshold = 80% of my min = {threshold}")
    else:
        # No friend data yet — estimate conservatively from own scores
        threshold = int(min_me * 0.75)
        max_friend = 0
        friend_scores = []
        print(f"  No friend data → estimated threshold = 75% of my min = {threshold}")

    print(f"\n  >>> Threshold = {threshold} <<<")
    print(f"{'='*52}")

    # ── Save threshold ─────────────────────────────────────────────────────────
    os.makedirs("data/templates", exist_ok=True)
    with open("data/templates/threshold.txt", 'w') as f:
        f.write(str(threshold))
    print(f"\n  Threshold saved → data/templates/threshold.txt")
    print(f"  match.py will use this automatically!\n")

    # ── Graph ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    n_me = len(me_scores)
    x_me = list(range(1, n_me + 1))

    ax.plot(x_me, me_scores,
            color='#555577', linewidth=1.4, zorder=2, alpha=0.6)
    ax.scatter(x_me, me_scores,
               color='#00e676', s=100, marker='o', zorder=4,
               label='Me (authorized)')

    if friend_scores:
        n_fr = len(friend_scores)
        x_fr = list(range(n_me + 1, n_me + n_fr + 1))
        ax.plot(x_fr, friend_scores,
                color='#774444', linewidth=1.4, zorder=2, alpha=0.6)
        ax.scatter(x_fr, friend_scores,
                   color='#ff5252', s=100, marker='x', zorder=4,
                   label='Friend/Impostor')

    ax.axhline(y=threshold, color='white', linestyle='--',
               linewidth=2, zorder=5,
               label=f'Unlock Threshold = {threshold}')
    ax.axhline(y=min_me, color='#ffab00', linestyle=':',
               linewidth=1.5, zorder=5,
               label=f'My min score = {min_me}')

    top = max(me_scores) + 80
    ax.fill_between(range(0, n_me + (len(friend_scores) or 1) + 2),
                    threshold, top, color='#00e676', alpha=0.05)
    ax.fill_between(range(0, n_me + (len(friend_scores) or 1) + 2),
                    0, threshold, color='#ff1744', alpha=0.04)

    me_correct = sum(1 for s in me_scores if s >= threshold)
    accuracy   = (me_correct / n_me) * 100 if n_me else 0

    ax.text(0.98, 0.97,
            f"My Accuracy  : {accuracy:.1f}%\n"
            f"Granted      : {me_correct}/{n_me}\n"
            f"Threshold    : {threshold}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#111122',
                      edgecolor='#00e676', alpha=0.95))

    ax.tick_params(colors='#cccccc', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.set_xlim(0, n_me + len(friend_scores) + 2)
    ax.set_ylim(0, top + 40)
    ax.set_xlabel("Test Image Number", fontsize=11, color='#aaaacc', labelpad=10)
    ax.set_ylabel("Match Score", fontsize=11, color='#aaaacc', labelpad=10)
    ax.set_title(
        "BioRhythm Lock — Wrist Verification Results\n"
        "🟢 Green = Me (authorized)   🔴 Red = Friend (impostor)   — = Threshold",
        fontsize=13, color='white', pad=15, fontweight='bold')
    ax.yaxis.grid(True, color='#222244', linewidth=0.8)
    ax.xaxis.grid(True, color='#222244', linewidth=0.8)
    ax.legend(fontsize=9, loc='upper left',
              facecolor='#1a1a2e', edgecolor='#333366',
              labelcolor='white', framealpha=0.95)

    plt.tight_layout()
    plt.show()
