"""
Experiment 04 — Incremental pseudocode extraction (multi-turn, no Python).

Key differences from exp01-03:
  - Frames are sent in small batches (configurable batch_size)
  - Multi-turn conversation: LLM proposes → revises → finalises hypothesis
  - Output is PSEUDOCODE ONLY — no Python code generation
  - Supports loading external PNG frames (real game footage)
  - Verification still runs as before (second LLM checks pseudocode)

Modes:
  MODE = "magnetworld"   — use MagnetWorld episodes (default)
  MODE = "external"      — load PNGs from EXTERNAL_FRAMES_DIR

Run from project root:
  python experiments/exp04_incremental.py

Required env var:
  FAL_KEY = your_fal_api_key
"""

import os
import sys
import json
import copy
import glob
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vlm.extractor import (
    extract_rule_incremental, verify_pseudocode, save_frames_as_images, record_episode_gif
)

FRAMES_DIR  = "frames"
RESULTS_DIR = "results"
GIF_DIR     = "gifs"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Which mode to run
MODE = "magnetworld"          # "magnetworld" or "external"

# For external mode: directory containing numbered PNGs (frame_000.png, frame_001.png, ...)
EXTERNAL_FRAMES_DIR = "frames/external"

# Incremental settings
BATCH_SIZE = 3                # frames per batch sent to LLM
PROVIDER   = "fal"
MODEL      = "google/gemini-2.5-pro"
VERIFY_MODEL = "google/gemini-2.5-flash"

# Which MagnetWorld episode to use (only for magnetworld mode)
PRIMARY_EPISODE = "full_complex"


# -----------------------------------------------------------------------
# MagnetWorld episodes (same as exp02, imported here for convenience)
# -----------------------------------------------------------------------

def get_magnetworld_episodes():
    from magnet_env.magnet_world import EMPTY, WALL, METAL, AGENT, ICE, HOLE
    W, E, M, A, I, H = WALL, EMPTY, METAL, AGENT, ICE, HOLE

    return {
        "ice_demo": {
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, M, E, E, E, E, E, E, E, W],
                [W, E, A, I, I, I, I, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "actions": [3, 0, 2],
            "description": "Agent slides across ice strip, then normal moves",
        },

        "hole_demo": {
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, A, E, E, E, M, E, H, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "actions": [3, 3, 3, 3],
            "description": "Agent attracts metal into hole — both vanish",
        },

        "attract_and_block": {
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, A, E, M, W, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "actions": [3, 2, 0, 1],
            "description": "Blocked attraction (metal at wall), then normal movement",
        },

        "full_complex": {
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, A, E, M, E, H, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, I, I, I, I, I, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, W, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "actions": [
                3, 3,
                3, 3, 3,
                1, 1,
                3, 3,
                0, 0, 0, 0,
                2, 2, 2, 2,
                1, 1,
                2,
                1, 1, 1,
                3, 3, 3, 3,
                0, 0,
                2, 2, 2,
                0, 0,
            ],
            "description": "Hole consumption, ice slides, normal movement, internal walls",
        },
    }


def load_external_frames(frames_dir):
    """
    Load PNG frames from a directory. Expects files sorted by name:
      frame_000.png, frame_001.png, ...  OR  0.png, 1.png, ...
    Any naming that sorts correctly in lexicographic order works.
    """
    patterns = [
        os.path.join(frames_dir, "*.png"),
        os.path.join(frames_dir, "*.jpg"),
        os.path.join(frames_dir, "*.jpeg"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths.sort()
    if not paths:
        raise FileNotFoundError(
            f"No image files found in {frames_dir}. "
            f"Place numbered PNGs (frame_000.png, frame_001.png, ...) in the directory."
        )
    return paths


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Experiment 04 — Incremental Pseudocode Extraction")
    print(f"Mode: {MODE} | Batch size: {BATCH_SIZE}")
    print(f"Provider: {PROVIDER} / Model: {MODEL}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Get frames
    # ----------------------------------------------------------------
    if MODE == "external":
        print(f"\n[1/3] Loading external frames from {EXTERNAL_FRAMES_DIR}...")
        frame_paths = load_external_frames(EXTERNAL_FRAMES_DIR)
        episode_name = os.path.basename(EXTERNAL_FRAMES_DIR)
        print(f"  Found {len(frame_paths)} frames")
    else:
        print(f"\n[1/3] Generating MagnetWorld frames...")
        from magnet_env.magnet_world import MagnetWorld, ACTION_NAMES
        episodes = get_magnetworld_episodes()
        ep = episodes[PRIMARY_EPISODE]
        episode_name = PRIMARY_EPISODE

        env = MagnetWorld(copy.deepcopy(ep["grid"]))
        frame_paths = save_frames_as_images(
            env, ep["actions"],
            output_dir=os.path.join(FRAMES_DIR, f"exp04_{PRIMARY_EPISODE}"),
            prefix="frame",
            reset_grid=copy.deepcopy(ep["grid"])
        )

        # Also save GIF for inspection
        env2 = MagnetWorld(copy.deepcopy(ep["grid"]))
        record_episode_gif(
            env2, ep["actions"],
            output_path=os.path.join(GIF_DIR, f"exp04_{PRIMARY_EPISODE}.gif"),
            frame_duration_ms=700,
            reset_grid=copy.deepcopy(ep["grid"])
        )
        act_names = [ACTION_NAMES[a] for a in ep["actions"]]
        print(f"  Episode '{PRIMARY_EPISODE}': {len(frame_paths)} frames, actions={act_names}")

    # ----------------------------------------------------------------
    # Step 2: Incremental extraction
    # ----------------------------------------------------------------
    print(f"\n[2/3] Incremental extraction (batch_size={BATCH_SIZE})...")
    print(f"      {len(frame_paths)} frames → {-(-len(frame_paths) // BATCH_SIZE)} rounds\n")

    try:
        result = extract_rule_incremental(
            image_paths=frame_paths,
            batch_size=BATCH_SIZE,
            provider=PROVIDER,
            model=MODEL,
            verbose=True,
        )
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    pseudocode = result["pseudocode"]
    rounds = result["rounds"]

    # Save pseudocode
    pseudo_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp04_pseudocode.txt")
    with open(pseudo_path, "w") as f:
        f.write(f"Provider: {PROVIDER} / Model: {MODEL}\n")
        f.write(f"Mode: {MODE} | Episode: {episode_name}\n")
        f.write(f"Batch size: {BATCH_SIZE} | Rounds: {len(rounds)}\n")
        f.write(f"Total frames: {len(frame_paths)}\n\n")
        f.write("=" * 40 + "\n")
        f.write("FINAL PSEUDOCODE\n")
        f.write("=" * 40 + "\n\n")
        f.write(pseudocode)

    # Save round-by-round history
    rounds_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp04_rounds.txt")
    with open(rounds_path, "w") as f:
        for i, rnd in enumerate(rounds):
            f.write(f"\n{'=' * 40}\n")
            f.write(f"ROUND {i + 1} — Frames {rnd['batch_indices']}\n")
            f.write(f"{'=' * 40}\n\n")
            f.write(rnd["raw"])
            f.write("\n")

    print(f"\n  Pseudocode saved    → {pseudo_path}")
    print(f"  Round history saved → {rounds_path}")
    print(f"\n{'=' * 40}")
    print("FINAL PSEUDOCODE")
    print(f"{'=' * 40}\n")
    print(pseudocode)
    print(f"\n{'=' * 40}\n")

    # ----------------------------------------------------------------
    # Step 3: Verify pseudocode with second LLM
    # ----------------------------------------------------------------
    print(f"[3/3] Verifying pseudocode with {PROVIDER}/{VERIFY_MODEL}...")

    try:
        verification = verify_pseudocode(
            image_paths=frame_paths,
            pseudocode=pseudocode,
            provider=PROVIDER,
            model=VERIFY_MODEL,
        )
    except Exception as e:
        print(f"\n  Verification ERROR: {e}")
        verification = {"verdict": "ERROR", "issues": str(e),
                        "frame_analysis": "", "raw": ""}

    verify_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp04_verification.txt")
    with open(verify_path, "w") as f:
        f.write(f"Verifier: {PROVIDER} / {VERIFY_MODEL}\n")
        f.write(f"Verdict: {verification['verdict']}\n\n")
        f.write("=== FRAME-BY-FRAME ANALYSIS ===\n")
        f.write(verification['frame_analysis'] + "\n\n")
        f.write("=== ISSUES ===\n")
        f.write(verification['issues'] + "\n\n")
        f.write("=== FULL RAW RESPONSE ===\n")
        f.write(verification['raw'] + "\n")

    verdict = verification["verdict"]
    print(f"\n  Verdict: {verdict}")
    if verification["issues"] and verification["issues"].lower() != "none":
        print(f"  Issues:\n    {verification['issues']}")
    print(f"  Verification report → {verify_path}\n")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    summary = {
        "experiment": "exp04_incremental",
        "provider": PROVIDER,
        "model": MODEL,
        "mode": MODE,
        "episode": episode_name,
        "batch_size": BATCH_SIZE,
        "n_frames": len(frame_paths),
        "n_rounds": len(rounds),
        "verification_verdict": verdict,
        "verification_model": VERIFY_MODEL,
        "pseudocode_path": pseudo_path,
        "rounds_path": rounds_path,
        "verification_path": verify_path,
    }
    summary_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp04_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    run()
