"""
Experiment 02 — Complex MagnetWorld rule extraction (larger grids, ICE + HOLE).

Extends Experiment 01 with:
  - ICE cells: agent slides until hitting non-ice cell
  - HOLE cells: metal pushed into hole → both vanish
  - Larger 10x10 grids with internal walls

Run from project root:
  python experiments/exp02_complex_rule.py

Required env var:
  FAL_KEY = your_fal_api_key
"""

import os
import sys
import json
import copy
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from magnet_env.magnet_world import (
    MagnetWorld, EMPTY, WALL, METAL, ICE, HOLE, AGENT, ACTION_NAMES
)
from vlm.extractor import extract_rule, verify_pseudocode, save_frames_as_images, record_episode_gif
from eval.evaluator import build_complex_test_cases, evaluate_extracted_function, print_evaluation_report
from eval.visualizer import render_results

FRAMES_DIR  = "frames"
RESULTS_DIR = "results"
GIF_DIR     = "gifs"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Shorthand
W, E, M, A, I, H = WALL, EMPTY, METAL, AGENT, ICE, HOLE

# -----------------------------------------------------------------------
# Episode definitions — larger grids with ICE, HOLE, internal walls
# -----------------------------------------------------------------------

EPISODES = {

    # 10x10: demonstrates ICE sliding clearly (agent slides, metal untouched)
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
        # RIGHT (slide on ice to col 7, metal behind), UP (normal), LEFT (normal)
        "description": "Agent slides across ice strip, then normal moves — metal stays put",
    },

    # 10x10: demonstrates HOLE consuming metal
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
        # RIGHT x4: attract metal 3 times, then metal lands on hole and vanishes
        "description": "Agent attracts metal rightward until it falls into hole — both vanish",
    },

    # 10x10: attraction + wall block in corridor
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
        # RIGHT (toward metal, metal blocked by wall — cancel!),
        # LEFT (away — normal), UP (normal), DOWN (normal)
        "description": "Blocked attraction (metal at wall), then normal movement",
    },

    # 10x10: full showcase — ice slide, then attraction, then hole
    "full_complex": {
        "grid": [
            [W, W, W, W, W, W, W, W, W, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, M, E, E, E, E, E, E, E, W],
            [W, E, A, I, I, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, W, W, W, W, W, W, W, W, W],
        ],
        "actions": [3, 1, 1, 1, 2, 0, 0],
        # RIGHT (slide on ice to col 5, metal behind not toward),
        # DOWN x3 (normal, metal above not toward),
        # LEFT (normal), UP x2 (toward metal — attraction)
        "description": "Ice slide, normal movement, then attraction toward metal upward",
    },
}

PRIMARY_EPISODE = "full_complex"

PROVIDER = "fal"
MODEL    = "google/gemini-2.5-pro"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Experiment 02 — Complex MagnetWorld Rule Extraction")
    print(f"Provider: {PROVIDER} / Model: {MODEL}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Generate frames and GIFs
    # ----------------------------------------------------------------
    print("\n[1/3] Generating frames and GIFs...")

    episode_frames = {}
    for name, ep in EPISODES.items():
        grid = ep["grid"]
        acts = ep["actions"]
        act_names = [ACTION_NAMES[a] for a in acts]

        env = MagnetWorld(copy.deepcopy(grid))
        frame_paths = save_frames_as_images(
            env, acts,
            output_dir=os.path.join(FRAMES_DIR, f"exp02_{name}"),
            prefix="frame",
            reset_grid=copy.deepcopy(grid)
        )
        episode_frames[name] = (frame_paths, act_names)

        env2 = MagnetWorld(copy.deepcopy(grid))
        record_episode_gif(
            env2, acts,
            output_path=os.path.join(GIF_DIR, f"exp02_{name}.gif"),
            frame_duration_ms=700,
            reset_grid=copy.deepcopy(grid)
        )
        print(f"  Episode '{name}': {len(frame_paths)} frames, actions={act_names}")

    print(f"\n  GIFs saved to ./{GIF_DIR}/")

    # ----------------------------------------------------------------
    # Step 2: Call VLM
    # ----------------------------------------------------------------
    print(f"\n[2/3] Sending '{PRIMARY_EPISODE}' to {PROVIDER}/{MODEL}...")
    frame_paths, act_names = episode_frames[PRIMARY_EPISODE]
    print(f"      ({len(frame_paths)} frames in video-like sequence mode)")

    try:
        result = extract_rule(
            image_paths=frame_paths,
            action_name="",
            provider=PROVIDER,
            model=MODEL,
            is_sequence=True,
            action_sequence=act_names
        )
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("\n  Check that FAL_KEY is set in your environment variables.")
        return

    extracted_code = result["python"]
    pseudocode = result["pseudocode"]

    code_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp02_extracted.py")
    with open(code_path, "w") as f:
        f.write(f"# Provider: {PROVIDER} / Model: {MODEL}\n")
        f.write(f"# Episode: {PRIMARY_EPISODE}\n")
        f.write(f"# Actions: {act_names}\n\n")
        f.write(extracted_code)

    pseudo_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp02_pseudocode.txt")
    with open(pseudo_path, "w") as f:
        f.write(f"Provider: {PROVIDER} / Model: {MODEL}\n")
        f.write(f"Episode: {PRIMARY_EPISODE}\n")
        f.write(f"Actions: {act_names}\n\n")
        f.write(pseudocode)

    print(f"\n  Extracted code saved → {code_path}")
    print(f"  Pseudocode saved    → {pseudo_path}")
    print("\n--- Pseudocode ---")
    print(pseudocode)
    print("\n--- Extracted function ---")
    print(extracted_code)
    print("--- End ---\n")

    # ----------------------------------------------------------------
    # Step 3: Verify pseudocode with a second LLM call
    # ----------------------------------------------------------------
    VERIFY_MODEL = "google/gemini-2.5-flash"  # use a different model for verification
    print(f"[3/5] Verifying pseudocode with {PROVIDER}/{VERIFY_MODEL}...")
    print(f"      (second LLM checks if pseudocode matches observed transitions)")

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

    # Save verification report
    verify_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp02_verification.txt")
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
    # Step 4: Evaluate extracted code against ground truth
    # ----------------------------------------------------------------
    print("[4/5] Evaluating extracted function against full complex test suite...")
    test_cases = build_complex_test_cases()
    eval_result = evaluate_extracted_function(extracted_code, test_cases)
    print_evaluation_report(eval_result, provider=f"{PROVIDER}/{MODEL}")

    # ----------------------------------------------------------------
    # Step 5: Visualize results
    # ----------------------------------------------------------------
    vis_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp02_visual.png")
    render_results(eval_result, test_cases, vis_path, title=f"EXP02 — {PROVIDER}/{MODEL}")

    summary = {
        "experiment": "exp02_complex",
        "provider": PROVIDER,
        "model": MODEL,
        "episode": PRIMARY_EPISODE,
        "accuracy": eval_result["accuracy"],
        "n_passed": eval_result.get("n_passed", 0),
        "n_total": eval_result.get("n_total", 0),
        "verification_verdict": verdict,
        "verification_model": VERIFY_MODEL,
        "extracted_code_path": code_path,
        "pseudocode_path": pseudo_path,
        "verification_path": verify_path,
    }
    summary_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp02_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    run()
