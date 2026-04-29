"""
Experiment 05 — Incremental pseudocode extraction from real-world video frames.

Three real-world physics scenes (frames pre-extracted at 2 fps via ffmpeg):
  bouncing_ball   — slow-motion ball bouncing on a hard floor
  newtons_cradle  — Newton's cradle momentum transfer
  pendulum        — slow-motion pendulum swing

Videos downloaded from YouTube; frames live in frames/realworld/<scene>/.
Same incremental multi-turn approach as exp04 (pseudocode only, no Python).

To add a new scene:
  1. Download a video and run:
       ffmpeg -i your_video.mp4 -vf "fps=2,scale=640:-1" -q:v 2 frames/realworld/<scene>/frame_%03d.png
  2. Add an entry to SCENES below.

Run from project root:
  python experiments/exp05_realworld.py

Required env var:
  FAL_KEY = your_fal_api_key
"""

import os
import sys
import json
import glob
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vlm.extractor import extract_rule_incremental, verify_pseudocode

FRAMES_ROOT = "frames/realworld"
RESULTS_DIR = "results/exp05"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PROVIDER     = "fal"
MODEL        = "google/gemini-2.5-pro"
VERIFY_MODEL = "google/gemini-2.5-flash"
BATCH_SIZE   = 3

# Add / remove scenes here. Key = subdirectory under frames/realworld/.
SCENES = {
    "bouncing_ball":   "A ball bouncing on a hard floor filmed in slow motion.",
    "newtons_cradle":  "A Newton's cradle — metal balls suspended on strings, one side struck.",
    "pendulum":        "A pendulum swinging back and forth in slow motion.",
    "double_pendulum": "A double pendulum — two pendulums connected in series, exhibiting chaotic motion.",
    "cymatics":        "Sand on a metal plate driven by a speaker — Chladni patterns forming at resonant frequencies.",
    "metronomes":      "Multiple mechanical metronomes on a common movable platform, spontaneously synchronising.",
}


def load_frames(scene_name: str) -> list:
    d = os.path.join(FRAMES_ROOT, scene_name)
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {d}")
    return paths


def run_scene(scene_name: str, description: str) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Scene : {scene_name}")
    print(f"Desc  : {description}")
    print(f"{'=' * 60}")

    try:
        frame_paths = load_frames(scene_name)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    n_rounds = math.ceil(len(frame_paths) / BATCH_SIZE)
    print(f"  {len(frame_paths)} frames | batch={BATCH_SIZE} | ~{n_rounds} rounds\n")

    # ── incremental extraction ────────────────────────────────────────────────
    try:
        result = extract_rule_incremental(
            image_paths=frame_paths,
            batch_size=BATCH_SIZE,
            provider=PROVIDER,
            model=MODEL,
            verbose=True,
        )
    except Exception as e:
        print(f"  EXTRACTION ERROR: {e}")
        import traceback; traceback.print_exc()
        return None

    pseudocode = result["pseudocode"]
    rounds     = result["rounds"]

    pseudo_path = os.path.join(RESULTS_DIR, f"{scene_name}_{TIMESTAMP}_pseudocode.txt")
    with open(pseudo_path, "w") as f:
        f.write(f"Scene        : {scene_name}\n")
        f.write(f"Description  : {description}\n")
        f.write(f"Provider     : {PROVIDER} / {MODEL}\n")
        f.write(f"Frames       : {len(frame_paths)} | Batch : {BATCH_SIZE} | Rounds : {len(rounds)}\n\n")
        f.write("=" * 50 + "\nFINAL PSEUDOCODE\n" + "=" * 50 + "\n\n")
        f.write(pseudocode)

    rounds_path = os.path.join(RESULTS_DIR, f"{scene_name}_{TIMESTAMP}_rounds.txt")
    with open(rounds_path, "w") as f:
        for i, rnd in enumerate(rounds):
            label = "FINAL" if i == len(rounds) - 1 else f"Round {i + 1}"
            f.write(f"\n{'=' * 50}\n{label}  —  frames {rnd['batch_indices']}\n{'=' * 50}\n\n")
            f.write(rnd["raw"] + "\n")

    print(f"\n--- FINAL PSEUDOCODE ---\n{pseudocode}\n{'- ' * 20}\n")
    print(f"  Pseudocode    → {pseudo_path}")
    print(f"  Round history → {rounds_path}")

    # ── verification ─────────────────────────────────────────────────────────
    print(f"\n  Verifying with {VERIFY_MODEL}...")
    try:
        verification = verify_pseudocode(
            image_paths=frame_paths,
            pseudocode=pseudocode,
            provider=PROVIDER,
            model=VERIFY_MODEL,
        )
    except Exception as e:
        print(f"  VERIFICATION ERROR: {e}")
        verification = {"verdict": "ERROR", "issues": str(e),
                        "frame_analysis": "", "raw": ""}

    verify_path = os.path.join(RESULTS_DIR, f"{scene_name}_{TIMESTAMP}_verification.txt")
    with open(verify_path, "w") as f:
        f.write(f"Verifier : {PROVIDER} / {VERIFY_MODEL}\n")
        f.write(f"Verdict  : {verification['verdict']}\n\n")
        f.write("=== FRAME-BY-FRAME ANALYSIS ===\n")
        f.write(verification["frame_analysis"] + "\n\n")
        f.write("=== ISSUES ===\n")
        f.write(verification["issues"] + "\n\n")
        f.write("=== FULL RAW RESPONSE ===\n")
        f.write(verification["raw"] + "\n")

    verdict = verification["verdict"]
    print(f"  Verdict       : {verdict}")
    if verification["issues"] and verification["issues"].lower() not in ("", "none"):
        print(f"  Issues        : {verification['issues'][:400]}")
    print(f"  Verification  → {verify_path}")

    summary = {
        "experiment"           : "exp05_realworld",
        "scene"                : scene_name,
        "description"          : description,
        "provider"             : PROVIDER,
        "model"                : MODEL,
        "batch_size"           : BATCH_SIZE,
        "n_frames"             : len(frame_paths),
        "n_rounds"             : len(rounds),
        "verification_verdict" : verdict,
        "verification_model"   : VERIFY_MODEL,
        "pseudocode_path"      : pseudo_path,
        "rounds_path"          : rounds_path,
        "verification_path"    : verify_path,
    }
    summary_path = os.path.join(RESULTS_DIR, f"{scene_name}_{TIMESTAMP}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary       → {summary_path}")

    return summary


def run():
    print("=" * 60)
    print("Experiment 05 — Real-World Video Pseudocode Extraction")
    print(f"Provider : {PROVIDER} / {MODEL}  |  Batch size : {BATCH_SIZE}")
    print("=" * 60)

    all_results = []
    for scene_name, description in SCENES.items():
        summary = run_scene(scene_name, description)
        if summary:
            all_results.append(summary)

    print("\n" + "=" * 60)
    print("ALL SCENES COMPLETE")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['scene']:20s}  {r['n_frames']:3d} frames  "
              f"{r['n_rounds']:2d} rounds  verdict={r['verification_verdict']}")


if __name__ == "__main__":
    run()
