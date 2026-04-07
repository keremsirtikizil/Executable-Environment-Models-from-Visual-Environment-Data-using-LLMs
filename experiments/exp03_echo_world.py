"""
Experiment 03 — EchoWorld rule extraction.

A completely different game from MagnetWorld:
  - Echo moves OPPOSITE to agent
  - Blocked echo does NOT cancel agent's move
  - Void consumes echo (both vanish)
  - Beacon bounces agent one extra step

Run from project root:
  python experiments/exp03_echo_world.py

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

from echo_env.echo_world import (
    EchoWorld, EMPTY, WALL, ECHO, VOID, BEACON, AGENT, ACTION_NAMES
)
from vlm.extractor import extract_rule, verify_pseudocode, save_frames_as_images, record_episode_gif
from eval.echo_evaluator import build_echo_test_cases, evaluate_echo_function, print_echo_evaluation_report

FRAMES_DIR  = "frames"
RESULTS_DIR = "results"
GIF_DIR     = "gifs"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Shorthand
W, E, O, V, B, A = WALL, EMPTY, ECHO, VOID, BEACON, AGENT

# -----------------------------------------------------------------------
# Episode definitions
# -----------------------------------------------------------------------

EPISODES = {

    # Shows echo moving opposite in open space
    "echo_basics": {
        "grid": [
            [W, W, W, W, W, W, W, W, W, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, O, E, E, A, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, W, W, W, W, W, W, W, W, W],
        ],
        "actions": [3, 3, 1, 2, 0],
        "description": "Echo moves opposite — RIGHT/RIGHT/DOWN/LEFT/UP",
    },

    # Echo blocked by walls, agent still moves
    "echo_blocked": {
        "grid": [
            [W, W, W, W, W, W, W, W, W, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, O, E, E, E, A, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, W, W, W, W, W, W, W, W, W],
        ],
        "actions": [3, 3, 3],
        # Agent RIGHT x3: echo at (3,1) moves LEFT each time → hits wall at (3,0), stays
        "description": "Echo pushed against left wall — stays put, agent keeps moving",
    },

    # Void consumes echo
    "void_demo": {
        "grid": [
            [W, W, W, W, W, W, W, W, W, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, V, E, O, E, A, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, W, W, W, W, W, W, W, W, W],
        ],
        "actions": [3, 3],
        # RIGHT: echo (3,4) → LEFT → (3,3). RIGHT: echo (3,3) → LEFT → (3,2)=VOID → consumed!
        "description": "Echo pushed into void — both vanish",
    },

    # Beacon bounce
    "beacon_demo": {
        "grid": [
            [W, W, W, W, W, W, W, W, W, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, O, E, E, A, E, B, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, W, W, W, W, W, W, W, W, W],
        ],
        "actions": [3, 3],
        # RIGHT: agent (3,4)→(3,5), echo (3,1)→LEFT→(3,0)=W→stays.
        # RIGHT: agent (3,5)→(3,6)=BEACON, bounce→(3,7). Echo stays at (3,1).
        "description": "Agent hits beacon and bounces an extra cell",
    },

    # Full showcase: echo opposite + blocked + void + beacon
    "full_echo": {
        "grid": [
            [W, W, W, W, W, W, W, W, W, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, V, E, O, E, A, B, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, E, E, E, E, E, E, E, E, W],
            [W, W, W, W, W, W, W, W, W, W],
        ],
        "actions": [3, 1, 2, 2, 2],
        # RIGHT: agent (3,6)→(3,7)=BEACON, bounce→(3,8). Echo (3,4)→LEFT→(3,3).
        # DOWN: agent→(4,8). Echo (3,3)→UP→(2,3).
        # LEFT: agent→(4,7). Echo (2,3)→RIGHT→(2,4).
        # LEFT: agent→(4,6). Echo (2,4)→RIGHT→(2,5).
        # LEFT: agent→(4,5). Echo (2,5)→RIGHT→(2,6).
        "description": "Full showcase — beacon bounce, echo opposite in multiple directions",
    },
}

PRIMARY_EPISODE = "full_echo"

PROVIDER = "fal"
MODEL    = "google/gemini-2.5-pro"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Experiment 03 — EchoWorld Rule Extraction")
    print(f"Provider: {PROVIDER} / Model: {MODEL}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Generate frames and GIFs
    # ----------------------------------------------------------------
    print("\n[1/5] Generating frames and GIFs...")

    episode_frames = {}
    for name, ep in EPISODES.items():
        grid = ep["grid"]
        acts = ep["actions"]
        act_names = [ACTION_NAMES[a] for a in acts]

        env = EchoWorld(copy.deepcopy(grid))
        frame_paths = save_frames_as_images(
            env, acts,
            output_dir=os.path.join(FRAMES_DIR, f"exp03_{name}"),
            prefix="frame",
            reset_grid=copy.deepcopy(grid)
        )
        episode_frames[name] = (frame_paths, act_names)

        env2 = EchoWorld(copy.deepcopy(grid))
        record_episode_gif(
            env2, acts,
            output_path=os.path.join(GIF_DIR, f"exp03_{name}.gif"),
            frame_duration_ms=700,
            reset_grid=copy.deepcopy(grid)
        )
        print(f"  Episode '{name}': {len(frame_paths)} frames, actions={act_names}")

    print(f"\n  GIFs saved to ./{GIF_DIR}/")

    # ----------------------------------------------------------------
    # Step 2: Call VLM
    # ----------------------------------------------------------------
    print(f"\n[2/5] Sending '{PRIMARY_EPISODE}' to {PROVIDER}/{MODEL}...")
    frame_paths, act_names = episode_frames[PRIMARY_EPISODE]
    print(f"      ({len(frame_paths)} frames)")

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

    code_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp03_extracted.py")
    with open(code_path, "w") as f:
        f.write(f"# Provider: {PROVIDER} / Model: {MODEL}\n")
        f.write(f"# Episode: {PRIMARY_EPISODE}\n")
        f.write(f"# Actions: {act_names}\n\n")
        f.write(extracted_code)

    pseudo_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp03_pseudocode.txt")
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
    # Step 3: Verify pseudocode
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

    verify_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp03_verification.txt")
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
    # Step 4: Evaluate
    # ----------------------------------------------------------------
    print("[4/5] Evaluating extracted function against EchoWorld test suite...")
    test_cases = build_echo_test_cases()
    eval_result = evaluate_echo_function(extracted_code, test_cases)
    print_echo_evaluation_report(eval_result, provider=f"{PROVIDER}/{MODEL}")

    # ----------------------------------------------------------------
    # Step 5: Summary
    # ----------------------------------------------------------------
    summary = {
        "experiment": "exp03_echo_world",
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
    summary_path = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_exp03_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    run()
