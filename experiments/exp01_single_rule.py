"""
Experiment 01 — MagnetWorld rule extraction via fal.ai + Gemini 2.5 Flash.

What this does:
  1. Generates MagnetWorld observation frames (and a GIF for visual inspection)
  2. Sends them to Gemini 2.5 Flash via fal.ai in "video-like" sequence mode
  3. Evaluates the extracted Python function against ground truth test cases
  4. Saves results to results/

Run from project root:
  python experiments/exp01_single_rule.py

Required env var (set in PyCharm Run Config > Environment Variables):
  FAL_KEY = your_fal_api_key

Get your key at: https://fal.ai/dashboard
"""

import os
import sys
import json
import copy
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from magnet_env.magnet_world import MagnetWorld, EMPTY, WALL, METAL, AGENT, ACTION_NAMES
from vlm.extractor import extract_rule, verify_pseudocode, save_frames_as_images, record_episode_gif
from eval.evaluator   import build_test_cases, evaluate_extracted_function, print_evaluation_report
from eval.visualizer  import render_results

FRAMES_DIR = "frames"
RESULTS_DIR = "results"
GIF_DIR     = "gifs"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------------------------------------------------------
# Episode definitions
# Each episode = a starting grid + sequence of actions to demonstrate rules
# -----------------------------------------------------------------------

EPISODES = {

    # Extended sequence (~5x frames): drags metal across grid using attraction,
    # demonstrates wall-blocked attraction, mixes toward / away / orthogonal moves.
    "attract_3steps": {
        "grid": [
            [WALL, WALL,  WALL,  WALL,  WALL,  WALL,  WALL,  WALL],
            [WALL, AGENT, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, EMPTY, METAL, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, WALL,  WALL,  WALL,  WALL,  WALL,  WALL,  WALL],
        ],
        "actions": [1, 1, 1, 3, 0, 3, 3, 0, 2, 2, 2, 3, 2],
        # DOWN x2 (drag metal down), DOWN (blocked attraction — no-op),
        # RIGHT (drag), UP (away — agent only), RIGHT x2 (drag metal east),
        # UP (away), LEFT x3 (orthogonal — metal stationary), RIGHT (drag), LEFT (away).
        "description": "Extended attraction sequence — toward/away/blocked/orthogonal mixed",
    },

    # Mixed sequence: toward + away + orthogonal — tests all three rule cases
    "mixed_6steps": {
        "grid": [
            [WALL,  WALL,  WALL,  WALL,  WALL,  WALL],
            [WALL,  EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL,  EMPTY, AGENT, EMPTY, EMPTY, WALL],
            [WALL,  EMPTY, EMPTY, EMPTY, METAL, WALL],
            [WALL,  EMPTY, EMPTY, EMPTY, EMPTY, WALL],
            [WALL,  WALL,  WALL,  WALL,  WALL,  WALL],
        ],
        "actions": [3, 3, 0, 0, 2, 1],  # RIGHT, RIGHT, UP, UP, LEFT, DOWN
        "description": "Mixed directions — shows when metal moves and when it doesn't",
    },

    # Wall-blocking: metal against wall, agent moves toward it
    "wall_block": {
        "grid": [
            [WALL,  WALL,  WALL,  WALL,  WALL],
            [WALL,  EMPTY, EMPTY, EMPTY, WALL],
            [WALL,  AGENT, EMPTY, METAL, WALL],
            [WALL,  EMPTY, EMPTY, EMPTY, WALL],
            [WALL,  WALL,  WALL,  WALL,  WALL],
        ],
        "actions": [3, 3],   # RIGHT x2 — first move: metal blocked by wall; second: agent still blocked
        "description": "Metal against wall — neither object moves when attracted move is blocked",
    },
}

# Which episode to use for the VLM call (richest signal for rule induction)
PRIMARY_EPISODE = "attract_3steps"

# Provider and model
PROVIDER = "fal"
MODEL    = "google/gemini-2.5-pro"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Experiment 01 — MagnetWorld Rule Extraction")
    print(f"Provider: {PROVIDER} / Model: {MODEL}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Generate frames and GIFs for all episodes
    # ----------------------------------------------------------------
    print("\n[1/3] Generating frames and GIFs...")

    episode_frames = {}
    for name, ep in EPISODES.items():
        grid  = ep["grid"]
        acts  = ep["actions"]
        act_names = [ACTION_NAMES[a] for a in acts]

        # Save individual PNG frames
        env = MagnetWorld(copy.deepcopy(grid))
        frame_paths = save_frames_as_images(
            env, acts,
            output_dir=os.path.join(FRAMES_DIR, name),
            prefix="frame",
            reset_grid=copy.deepcopy(grid)
        )
        episode_frames[name] = (frame_paths, act_names)

        # Save GIF
        env2 = MagnetWorld(copy.deepcopy(grid))
        record_episode_gif(
            env2, acts,
            output_path=os.path.join(GIF_DIR, f"{name}.gif"),
            frame_duration_ms=700,
            reset_grid=copy.deepcopy(grid)
        )
        print(f"  Episode '{name}': {len(frame_paths)} frames, actions={act_names}")

    print(f"\n  GIFs saved to ./{GIF_DIR}/ — open these to visually inspect episodes")

    # ----------------------------------------------------------------
    # Step 2: Call VLM with primary episode in sequence (video-like) mode
    # ----------------------------------------------------------------
    print(f"\n[2/3] Sending '{PRIMARY_EPISODE}' to {PROVIDER}/{MODEL}...")
    print(f"      ({len(episode_frames[PRIMARY_EPISODE][0])} frames in video-like sequence mode)")

    frame_paths, act_names = episode_frames[PRIMARY_EPISODE]

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

    # Save extracted Python code
    code_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_extracted.py")
    with open(code_path, "w") as f:
        f.write(f"# Provider: {PROVIDER} / Model: {MODEL}\n")
        f.write(f"# Episode: {PRIMARY_EPISODE}\n")
        f.write(f"# Actions: {act_names}\n\n")
        f.write(extracted_code)

    # Save pseudocode
    pseudo_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_pseudocode.txt")
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
    VERIFY_MODEL = "google/gemini-2.5-flash"
    print(f"[3/5] Verifying pseudocode with {PROVIDER}/{VERIFY_MODEL}...")

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

    verify_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_verification.txt")
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
    # Step 4: Evaluate against ground truth
    # ----------------------------------------------------------------
    print("[4/5] Evaluating extracted function...")
    test_cases = build_test_cases()
    eval_result = evaluate_extracted_function(extracted_code, test_cases)
    print_evaluation_report(eval_result, provider=f"{PROVIDER}/{MODEL}")

    # ----------------------------------------------------------------
    # Step 5: Visualize results
    # ----------------------------------------------------------------
    vis_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_visual.png")
    render_results(eval_result, test_cases, vis_path, title=f"{PROVIDER}/{MODEL}")

    # Save summary
    summary = {
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
    summary_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    run()