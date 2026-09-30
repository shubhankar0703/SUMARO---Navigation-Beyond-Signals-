"""
Generate presentation-ready PNG graph with 100% transparent background
for SUMARO Smart India Hackathon 2026 PPT.

Visualizes standardized 60-second GNSS blackout benchmark results on held-out IO-VNBD Session Vw1.
Target resolution: 1920 x 1080 px (16:9 widescreen presentation fit).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sumaro_vw1_outage_error_ppt.png")

# Set font family to Segoe UI with fallback to Arial
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#94a3b8"
plt.rcParams["axes.linewidth"] = 1.2

# Data (exact canonical Vw1 benchmark values)
estimators = [
    "EKF",
    "RBPF",
    "EKF + Real ML",
    "RBPF + Real ML",
]
errors = [
    9512.44,
    2404.97,
    9363.03,
    805.97,
]
value_labels = [
    "9,512 m",
    "2,405 m",
    "9,363 m",
    "806 m",
]

# Colors
# Neutral slates for baselines; vibrant SUMARO brand cyan/blue for primary RBPF + Real ML
bar_colors = [
    "#94a3b8",  # EKF - subtle slate
    "#64748b",  # RBPF - medium slate
    "#94a3b8",  # EKF + Real ML - subtle slate
    "#0284c7",  # RBPF + Real ML - Primary SUMARO vibrant cyan/blue
]

edge_colors = [
    "#64748b",
    "#475569",
    "#64748b",
    "#0369a1",
]

# Setup 1920 x 1080 figure (12.8 in x 7.2 in @ 150 DPI)
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Layout margins (leaves generous transparent padding around entire chart)
plt.subplots_adjust(left=0.24, right=0.88, top=0.80, bottom=0.18)

y_pos = np.arange(len(estimators))
bar_height = 0.50

# Draw horizontal bars
bars = ax.barh(
    y_pos,
    errors,
    height=bar_height,
    color=bar_colors,
    edgecolor=edge_colors,
    linewidth=1.5,
    zorder=3,
)

# Invert Y axis so order is top-to-bottom: EKF -> RBPF -> EKF + Real ML -> RBPF + Real ML
ax.invert_yaxis()

# X-axis configuration
ax.set_xlim(0, 11500)
ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
ax.set_xticklabels(["0 m", "2,000 m", "4,000 m", "6,000 m", "8,000 m", "10,000 m"], fontsize=14, color="#334155", weight="medium")

# Subtle vertical grid lines for readability
ax.grid(axis="x", color="#cbd5e1", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

# Style spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#64748b")
ax.spines["bottom"].set_linewidth(1.5)

# Y-axis ticks and labels
ax.set_yticks(y_pos)
y_tick_labels = ax.set_yticklabels(estimators, fontsize=18, weight="bold", color="#1e293b")

# Highlight "RBPF + Real ML" label on the Y-axis
y_tick_labels[3].set_color("#0284c7")
y_tick_labels[3].set_weight("extra bold")

# Remove tick marks
ax.tick_params(axis="both", length=0, pad=16)

# X-axis label
ax.set_xlabel("Maximum Position Error (m)", fontsize=16, weight="bold", color="#1e293b", labelpad=18)

# Add exact value labels beside each bar
for i, (bar, val_str) in enumerate(zip(bars, value_labels)):
    w = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2.0
    
    if i == 3:
        # Highlighted primary SUMARO configuration
        ax.text(
            w + 220,
            y,
            val_str,
            ha="left",
            va="center",
            fontsize=20,
            weight="extra bold",
            color="#0284c7",
            zorder=4,
        )
    else:
        ax.text(
            w + 220,
            y,
            val_str,
            ha="left",
            va="center",
            fontsize=17,
            weight="bold",
            color="#334155",
            zorder=4,
        )

# Main Title (crisp dark slate, legible on white and light backgrounds)
fig.text(
    0.24,
    0.91,
    "Position Error During GNSS Outage",
    fontsize=26,
    weight="extra bold",
    color="#0f172a",
    ha="left",
    va="top",
)

# Unobtrusive caption
fig.text(
    0.24,
    0.855,
    "Held-out Vw1 test • standardized 60-second GNSS blackout",
    fontsize=14,
    weight="normal",
    color="#64748b",
    ha="left",
    va="top",
)

# Save figure with 100% transparent background
plt.savefig(
    OUTPUT_PATH,
    transparent=True,
    dpi=150,
    facecolor="none",
    edgecolor="none",
)
plt.close()

# Verify file with Pillow
img = Image.open(OUTPUT_PATH)
w, h = img.size
mode = img.mode
has_alpha = "A" in mode
bands = img.getbands()

print(f"Graph generated successfully!")
print(f"Path: {OUTPUT_PATH}")
print(f"Dimensions: {w} x {h} px")
print(f"Color Mode: {mode} (Bands: {bands})")
print(f"Has Alpha Channel: {has_alpha}")
