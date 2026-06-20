"""
Generate a demo GIF showcasing the Bio-WM pipeline.

Renders a 4-panel dashboard:
  1. Ground truth (clean frame — white block only)
  2. Noisy 64x64 grid-world frame (with flicker noise)
  3. Digital latent grid (8x8 heatmap) with trajectory history
  4. Biological prediction rollout (H=8 imaginary future)

Usage:
    python generate_demo_gif.py
"""

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for frame saving
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

from env import ParticleBoxEnv
from encoder import ParticleEncoder, pretrain_encoder
from sigreg import SIGRegLoss
from bio_loop import BioWorldModelLoop, calculate_centroid
import cl_sdk as cl

# ── Configuration ────────────────────────────────────────────
TOTAL_STEPS   = 400        # Total sim steps to run
WARMUP_STEPS  = 50         # Steps before we start capturing frames
CAPTURE_EVERY = 3          # Capture a frame every N steps
GIF_FPS       = 12         # Frames per second in the output GIF
OUTPUT_GIF    = "demo.gif" # Output file name
FRAME_DIR     = "_gif_frames"
# ─────────────────────────────────────────────────────────────

# ── Style ────────────────────────────────────────────────────
DARK_BG     = "#0d1117"
PANEL_BG    = "#161b22"
ACCENT_CYAN = "#58a6ff"
ACCENT_RED  = "#f85149"
ACCENT_GOLD = "#e3b341"
TEXT_COLOR   = "#c9d1d9"
GRID_COLOR   = "#30363d"

rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "font.family":      "sans-serif",
    "font.size":        10,
})
# ─────────────────────────────────────────────────────────────


def render_frame(step, clean_frame, noisy_frame, z_next, actual_path, pred_future_centroids, save_path):
    """Render a single polished 4-panel dashboard frame."""
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.5),
                            gridspec_kw={"wspace": 0.25})
    fig.suptitle(
        f"Bio-WM  |  Closed-Loop World Model  |  Step {step}",
        fontsize=14, fontweight="bold", color=ACCENT_GOLD, y=0.97
    )

    # ── Panel 1: Ground Truth (clean frame) ──────────────────
    axs[0].imshow(clean_frame, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Ground Truth", fontsize=11, pad=8)
    axs[0].axis("off")
    for spine in axs[0].spines.values():
        spine.set_edgecolor(ACCENT_GOLD)
        spine.set_linewidth(1.2)

    # ── Panel 2: Noisy observation ───────────────────────────
    axs[1].imshow(noisy_frame, cmap="inferno", vmin=0, vmax=1)
    axs[1].set_title("Noisy Observation (10% flicker)", fontsize=11, pad=8)
    axs[1].axis("off")
    for spine in axs[1].spines.values():
        spine.set_edgecolor(ACCENT_CYAN)
        spine.set_linewidth(1.2)

    # ── Panel 3: Digital latent grid + history path ──────────
    grid_8x8 = z_next.reshape(8, 8)
    axs[2].imshow(grid_8x8, cmap="magma", interpolation="nearest",
                  extent=[0, 8, 8, 0], vmin=0)
    if len(actual_path) > 1:
        act = np.array(actual_path)
        axs[2].plot(act[:, 0], act[:, 1], color=ACCENT_CYAN,
                    linewidth=1.8, alpha=0.85, label="History")
        axs[2].scatter(act[-1, 0], act[-1, 1], color=ACCENT_CYAN,
                       edgecolors="white", s=60, zorder=5, label="Current Zt")
    axs[2].set_xlim(0, 8)
    axs[2].set_ylim(0, 8)
    axs[2].invert_yaxis()
    axs[2].set_title("Digital Latent Grid (Zt)", fontsize=11, pad=8)
    axs[2].legend(loc="upper right", fontsize=7, framealpha=0.6,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    axs[2].grid(True, linestyle="--", alpha=0.2, color=GRID_COLOR)

    # ── Panel 4: Biological prediction rollout ───────────────
    current_pos = actual_path[-1] if len(actual_path) > 0 else np.array([4.0, 4.0])
    pred_coords = np.array(pred_future_centroids)
    full_pred = np.vstack([current_pos, pred_coords])

    axs[3].plot(full_pred[:, 0], full_pred[:, 1], color=ACCENT_RED,
                linewidth=2.5, marker="o", markersize=5, label="Imagined Future (H=8)")
    if len(actual_path) >= 8:
        past = np.array(actual_path[-8:])
        axs[3].plot(past[:, 0], past[:, 1], color=ACCENT_CYAN,
                    linewidth=1.5, linestyle="--", alpha=0.6, label="Actual Past")
    axs[3].set_xlim(0, 8)
    axs[3].set_ylim(0, 8)
    axs[3].invert_yaxis()
    axs[3].set_title("BNN Prediction (H=8 Rollout)", fontsize=11, pad=8)
    axs[3].legend(loc="upper right", fontsize=7, framealpha=0.6,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    axs[3].grid(True, linestyle="--", alpha=0.25, color=GRID_COLOR)

    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.15)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare frame output directory
    os.makedirs(FRAME_DIR, exist_ok=True)

    # ── Initialize modules ───────────────────────────────────
    env     = ParticleBoxEnv()
    encoder = ParticleEncoder().to(device)
    sigreg  = SIGRegLoss(channels=64, sketch_dim=64).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=2e-4)
    criterion = nn.MSELoss()

    print("Pre-training CNN encoder...")
    pretrain_encoder(encoder, num_samples=800, epochs=15)

    replay = []

    with cl.open() as loop:
        bio_loop = BioWorldModelLoop(loop)
        noisy, clean, pos = env.reset()
        replay.append(noisy)

        actual_path = []
        action = np.random.choice([0, 1, 2, 3])
        frame_idx = 0

        print(f"Running {TOTAL_STEPS}-step simulation (capturing every {CAPTURE_EVERY} steps after {WARMUP_STEPS} warmup)...")

        for step in range(1, TOTAL_STEPS + 1):
            if step % 15 == 0:
                action = np.random.choice([0, 1, 2, 3])

            # Encode current
            s_curr = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                z_curr = torch.softmax(encoder(s_curr).squeeze(0), dim=0).cpu().numpy()

            # Step environment
            next_noisy, next_clean, next_pos, collision = env.step(action)
            replay.append(next_noisy)
            if len(replay) > 1000:
                replay.pop(0)

            # Encode next
            encoder.train()
            optimizer.zero_grad()
            s_next = torch.tensor(next_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            z_next_t = torch.softmax(encoder(s_next), dim=1)
            z_next = z_next_t.squeeze(0).detach().cpu().numpy()

            # BNN predict & learn
            c_act = calculate_centroid(z_next)
            y_target = np.array([c_act[0], c_act[1], env.dx / 8.0, env.dy / 8.0], dtype=np.float32)
            z_pred = bio_loop.predict_and_learn(z_curr, action, z_next, y_target, boundary_penalty=collision)

            # Loss
            z_pred_t = torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(device)
            loss_pred = criterion(z_next_t, z_pred_t)

            if len(replay) >= 64:
                idxs = np.random.choice(len(replay), 64, replace=False)
                batch = np.array([replay[i] for i in idxs], dtype=np.float32)
                batch_t = torch.tensor(batch).unsqueeze(1).to(device)
                z_batch = encoder(batch_t)
                loss_sig = sigreg(z_batch)
            else:
                loss_sig = torch.tensor(0.0, device=device)

            (loss_pred + 25.0 * loss_sig).backward()
            optimizer.step()

            # Tracking
            actual_path.append(c_act)
            if len(actual_path) > 30:
                actual_path.pop(0)

            noisy = next_noisy
            clean = next_clean

            # Trajectory rollout
            future_actions = [action] * 8
            vel = np.array([env.dx / 8.0, env.dy / 8.0], dtype=np.float32)
            z_traj = bio_loop.predict_trajectory(z_curr, future_actions, vel, horizon=8)
            pred_centroids = [calculate_centroid(z) for z in z_traj]

            # ── Capture frame ────────────────────────────────
            if step >= WARMUP_STEPS and step % CAPTURE_EVERY == 0:
                save_path = os.path.join(FRAME_DIR, f"frame_{frame_idx:04d}.png")
                render_frame(step, clean, noisy, z_next, actual_path, pred_centroids, save_path)
                frame_idx += 1
                if step % 50 == 0:
                    print(f"  Step {step}/{TOTAL_STEPS}  |  {frame_idx} frames captured")

        print(f"\n[OK] Captured {frame_idx} frames total.")

    # ── Compile GIF ──────────────────────────────────────────
    print(f"Compiling GIF at {GIF_FPS} fps...")
    frame_files = sorted(os.listdir(FRAME_DIR))
    frames = [Image.open(os.path.join(FRAME_DIR, f)) for f in frame_files if f.endswith(".png")]

    if frames:
        duration_ms = int(1000 / GIF_FPS)
        frames[0].save(
            OUTPUT_GIF,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        print(f"[OK] Saved {OUTPUT_GIF}  ({len(frames)} frames, {len(frames)/GIF_FPS:.1f}s)")
    else:
        print("[WARN] No frames captured!")

    # Cleanup temp frames
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    print("[OK] Cleaned up temporary frames.")


if __name__ == "__main__":
    main()
