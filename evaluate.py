# evaluate.py

import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from preprocess import preprocess_vein

ENROLL_COUNT = 15   # must match enroll.py


def get_score(image_path, template):
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
    all_f = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    return all_f[skip:]


if __name__ == "__main__":

    # Load me template
    try:
        with open("data/templates/me_template.pkl", 'rb') as f:
            template = pickle.load(f)
        print("\n  Template loaded successfully.")
    except FileNotFoundError:
        print("  ERROR: Template not found. Run enroll.py first!")
        exit()

    # me test images (authorized — should score HIGH)
    me_test = get_test_files(r"C:\Users\PRANAY\OneDrive\Desktop\Mini_project_1\test_images")

    # Also load stranger scores silently just to calculate threshold
    stranger_test = sorted([
        f for f in os.listdir(r"C:\Users\PRANAY\OneDrive\Desktop\Mini_project_1\test_images")
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\n  me test images  : {len(me_test)}")
    print(f"  Stranger images     : {len(stranger_test)} (used only for threshold calc)\n")

    # Score everything
    me_scores   = []
    stranger_scores = []

    print("  Scoring me images...")
    for f in me_test:
        s = get_score(f"data/me/{f}", template)
        me_scores.append(s)
        print(f"    {f:<30} →  {s:>4}  (authorized)")

    print("\n  Scoring stranger images (for threshold calculation only)...")
    for f in stranger_test:
        s = get_score(f"data/me/{f}", template)
        stranger_scores.append(s)
        print(f"    {f:<30} →  {s:>4}  (stranger)")

    # ── Threshold Calculation ─────────────────────────────────────────────────
    min_me   = min(me_scores)   if me_scores   else 0
    max_stranger = max(stranger_scores) if stranger_scores else 0
    gap          = min_me - max_stranger

    print(f"\n{'='*52}")
    print(f"  me scores   →  min={min(me_scores)}  "
          f"max={max(me_scores)}  "
          f"avg={np.mean(me_scores):.1f}")
    print(f"  Stranger scores →  min={min(stranger_scores)}  "
          f"max={max(stranger_scores)}  "
          f"avg={np.mean(stranger_scores):.1f}")
    print(f"  Gap (me min - stranger max) = {gap}")

    if gap > 0:
        threshold = int(max_stranger + gap * 0.5)
        print(f"\n  GAP EXISTS — system can tell them apart!")
        print(f"  Threshold = max_stranger + (gap × 0.5) = {threshold}")
    else:
        threshold = int(min_me * 0.80)
        print(f"\n  WARNING: Scores overlap by {abs(gap)} points.")
        print(f"  Using fallback threshold = 80% of me min = {threshold}")

    print(f"\n  >>> Update match.py:  THRESHOLD = {threshold} <<<")
    print(f"{'='*52}")

    # ── Graph (me Only) ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    n_me = len(me_scores)
    x_me = list(range(1, n_me + 1))

    # Connecting line
    ax.plot(x_me, me_scores,
            color='#555577', linewidth=1.4, zorder=2, alpha=0.6)

    # me dots — green circles
    ax.scatter(x_me, me_scores,
               color='#00e676', s=100, marker='o', zorder=4,
               label='me (authorized)')

    # Threshold line
    ax.axhline(y=threshold, color='white', linestyle='--',
               linewidth=2, zorder=5,
               label=f'Unlock Threshold = {threshold}')

    # Min me score line
    ax.axhline(y=min_me, color='#ffab00', linestyle=':',
               linewidth=1.5, zorder=5,
               label=f'me min score = {min_me}')

    # Max stranger score line (shown as reference for threshold explanation)
    ax.axhline(y=max_stranger, color='#ff6d00', linestyle=':',
               linewidth=1.5, zorder=5,
               label=f'Stranger max score = {max_stranger} (threshold reference)')

    # Shading — GRANTED zone (above threshold)
    top = max(me_scores) + 80
    ax.fill_between(x_me, threshold, top,
                    color='#00e676', alpha=0.06)

    # Shading — DENIED zone (below threshold)
    ax.fill_between(x_me, 0, threshold,
                    color='#ff1744', alpha=0.05)

    # GRANTED / DENIED zone labels
    ax.text(n_me * 0.5, threshold + 15,
            "✅  GRANTED ZONE  (above threshold)",
            ha='center', fontsize=10, color='#00e676', fontweight='bold')
    ax.text(n_me * 0.5, threshold - 25,
            "❌  DENIED ZONE  (below threshold)",
            ha='center', fontsize=10, color='#ff4444', fontweight='bold')

    # Gap annotation arrow
    if gap > 0:
        ax.annotate('',
            xy=(n_me + 0.5, threshold),
            xytext=(n_me + 0.5, min_me),
            arrowprops=dict(arrowstyle='<->', color='#ffab00', lw=2))
        ax.text(n_me + 0.7,
                (threshold + min_me) / 2,
                f'  Safety\n  gap\n  = {gap}',
                color='#ffab00', fontsize=9)

    # Accuracy for me only
    me_correct = sum(1 for s in me_scores if s >= threshold)
    accuracy       = (me_correct / n_me) * 100

    ax.text(0.98, 0.97,
            f"me Accuracy: {accuracy:.1f}%\n"
            f"me granted  : {me_correct}/{n_me}\n"
            f"Threshold used  : {threshold}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, color='white',
            bbox=dict(boxstyle='round,pad=0.6',
                      facecolor='#111122',
                      edgecolor='#00e676', alpha=0.95))

    # Threshold explanation box (bottom right)
    if gap > 0:
        thresh_explanation = (
            f"📐  HOW THRESHOLD = {threshold} WAS CALCULATED\n\n"
            f"  me min score   = {min_me}\n"
            f"  Stranger max score = {max_stranger}\n"
            f"  Gap                = {min_me} - {max_stranger} = {gap}  ✅ (no overlap)\n\n"
            f"  Threshold = Stranger max + (Gap × 0.5)\n"
            f"            = {max_stranger} + ({gap} × 0.5)\n"
            f"            = {threshold}\n\n"
            f"  This places the threshold exactly in the\n"
            f"  middle of the safety gap between me\n"
            f"  and stranger scores."
        )
    else:
        thresh_explanation = (
            f"📐  HOW THRESHOLD = {threshold} WAS CALCULATED\n\n"
            f"  me min score   = {min_me}\n"
            f"  Stranger max score = {max_stranger}\n"
            f"  Gap                = {gap}  ⚠️ (scores overlap!)\n\n"
            f"  Fallback: Threshold = me min × 0.80\n"
            f"          = {min_me} × 0.80 = {threshold}\n\n"
            f"  WARNING: Add more varied images to\n"
            f"  improve separation."
        )

    fig.text(0.60, 0.01, thresh_explanation,
             fontsize=9, color='#bbbbdd',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a0a1a',
                       edgecolor='#ffab00', alpha=0.95))

    # How to read box
    notes = (
        "📌  HOW TO READ THIS GRAPH\n\n"
        "• Each green dot = one me wrist image tested against the stored template\n"
        "• Y-axis = match score (how many vein feature points matched)\n"
        "• All dots should be ABOVE the white threshold line → GRANTED\n"
        "• White dashed line = unlock threshold (boundary between granted & denied)\n"
        "• Orange dotted line = me's min score  |  Red dotted = stranger's max score\n"
        "• Threshold sits in the middle of the gap between these two lines"
    )
    fig.text(0.01, 0.01, notes, fontsize=8.5, color='#bbbbdd',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a0a1a',
                       edgecolor='#333366', alpha=0.95))

    # Styling
    ax.tick_params(colors='#cccccc', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.set_xlim(0, n_me + 2)
    ax.set_ylim(0, top + 40)
    ax.set_xlabel("Test Image Number  (enrollment images not shown)",
                  fontsize=11, color='#aaaacc', labelpad=10)
    ax.set_ylabel("Match Score\n(how many vein features matched the template)",
                  fontsize=11, color='#aaaacc', labelpad=10)
    ax.set_title(
        "BioRhythm Lock — Authorized Wrist Verification\n"
        "🟢 Green circle = Authorized (me)     |     White dashed = Unlock Threshold",
        fontsize=13, color='white', pad=15, fontweight='bold')
    ax.yaxis.grid(True, color='#222244', linewidth=0.8)
    ax.xaxis.grid(True, color='#222244', linewidth=0.8)
    ax.legend(fontsize=9, loc='upper left',
              facecolor='#1a1a2e', edgecolor='#333366',
              labelcolor='white', framealpha=0.95)

    plt.tight_layout(rect=[0, 0.28, 1, 1])
    os.makedirs("data/templates", exist_ok=True)
    with open("data/templates/threshold.txt", 'w') as f:
        f.write(str(threshold))
    print(f"\n  Threshold saved to data/templates/threshold.txt")
    print(f"  match.py will now use this automatically!")
    plt.show()