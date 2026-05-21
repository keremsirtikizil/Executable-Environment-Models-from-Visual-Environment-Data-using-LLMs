"""
Experiment 06 — Similar-video comparison.

Runs PURE on paired physics videos that show the same phenomenon with small
variations, then uses a VLM to highlight what differs between the pairs.

Pairs:
  bowling_pins   vs  bowling_strike    (bowling — idle pins vs. live strike)
  billiard_break vs  billiards_physics (billiards — break shot vs. longer game)
  pendulum       vs  double_pendulum   (reuses latest exp05 results — no re-run)

Run from project root:
  python experiments/exp06_comparison.py

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
RESULTS_DIR = "results/exp06"
EXP05_DIR   = "results/exp05"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
PROVIDER     = "fal"
MODEL        = "google/gemini-2.5-pro"
VERIFY_MODEL = "google/gemini-2.5-flash"
BATCH_SIZE   = 3

# New scenes to run (both sides of each comparison pair)
SCENES = {
    "bowling_pins":    "Bowling pins standing on a lane — a ball rolls in and scatters the pins.",
    "bowling_strike":  "A bowling ball strikes all ten pins simultaneously — a perfect strike.",
    "billiard_break":  "A billiards break shot — cue ball struck hard, scattering the rack.",
    "billiards_physics": "Billiard balls rolling and colliding on a pool table — various shots.",
}

# Existing exp05 results to reuse for the pendulum pair (latest run each)
REUSE_PAIRS = [
    ("pendulum",        "double_pendulum"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_frames(scene_name: str) -> list:
    d = os.path.join(FRAMES_ROOT, scene_name)
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames in {d}")
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

    try:
        result = extract_rule_incremental(
            image_paths=frame_paths,
            batch_size=BATCH_SIZE,
            provider=PROVIDER,
            model=MODEL,
            verbose=True,
            prompt_style="physics",
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
    print(f"  Verdict : {verdict}")
    if verification["issues"] and verification["issues"].lower() not in ("", "none"):
        print(f"  Issues  : {verification['issues'][:300]}")

    summary = {
        "experiment"           : "exp06_comparison",
        "scene"                : scene_name,
        "description"          : description,
        "provider"             : PROVIDER,
        "model"                : MODEL,
        "batch_size"           : BATCH_SIZE,
        "n_frames"             : len(frame_paths),
        "n_rounds"             : len(rounds),
        "verification_verdict" : verdict,
        "pseudocode"           : pseudocode,
        "pseudocode_path"      : pseudo_path,
        "rounds_path"          : rounds_path,
        "verification_path"    : verify_path,
    }
    summary_path = os.path.join(RESULTS_DIR, f"{scene_name}_{TIMESTAMP}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "pseudocode"}, f, indent=2)

    return summary


# ── reuse existing exp05 pseudocode ───────────────────────────────────────────

def load_latest_exp05_pseudocode(scene_name: str) -> str | None:
    pattern = os.path.join(EXP05_DIR, f"{scene_name}_*_pseudocode.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    path = matches[-1]
    with open(path) as f:
        text = f.read()
    # extract the part after the header
    marker = "FINAL PSEUDOCODE\n" + "=" * 50
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text.strip()


# ── VLM comparison ────────────────────────────────────────────────────────────

def compare_pair(scene_a: str, pseudo_a: str, scene_b: str, pseudo_b: str) -> str:
    """Ask the VLM to compare two pseudocodes from similar scenes."""
    from openai import OpenAI

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise EnvironmentError("FAL_KEY not set.")

    client = OpenAI(
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="not-needed",
        default_headers={"Authorization": f"Key {fal_key}"},
    )

    prompt = (
        f"You are analysing the output of a physics-rule extraction system called PURE.\n\n"
        f"Below are two pseudocode summaries produced by PURE for two SIMILAR but DISTINCT "
        f"physics videos.\n\n"
        f"Scene A: {scene_a}\n"
        f"{'=' * 40}\n{pseudo_a}\n\n"
        f"Scene B: {scene_b}\n"
        f"{'=' * 40}\n{pseudo_b}\n\n"
        f"Please provide:\n"
        f"1. SHARED PHYSICS — rules / behaviours that appear in BOTH pseudocodes.\n"
        f"2. UNIQUE TO A   — physics or behaviours only captured for {scene_a}.\n"
        f"3. UNIQUE TO B   — physics or behaviours only captured for {scene_b}.\n"
        f"4. SENSITIVITY   — Does the difference in the videos (the 'small thing that changed') "
        f"show up clearly in the pseudocode difference? Rate: HIGH / MEDIUM / LOW and explain.\n"
        f"Be concise and precise. Use bullet points."
    )

    response = client.chat.completions.create(
        model=VERIFY_MODEL,
        max_tokens=4000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def run_comparison(scene_a: str, pseudo_a: str, scene_b: str, pseudo_b: str) -> None:
    print(f"\n{'#' * 60}")
    print(f"COMPARISON:  {scene_a}  vs  {scene_b}")
    print(f"{'#' * 60}")

    try:
        analysis = compare_pair(scene_a, pseudo_a, scene_b, pseudo_b)
    except Exception as e:
        analysis = f"Comparison error: {e}"

    print(analysis)

    cmp_path = os.path.join(
        RESULTS_DIR,
        f"compare_{scene_a}_vs_{scene_b}_{TIMESTAMP}.txt",
    )
    with open(cmp_path, "w") as f:
        f.write(f"Pair     : {scene_a}  vs  {scene_b}\n")
        f.write(f"Model    : {PROVIDER} / {VERIFY_MODEL}\n\n")
        f.write("=" * 50 + "\nCOMPARISON ANALYSIS\n" + "=" * 50 + "\n\n")
        f.write(analysis + "\n\n")
        f.write("=" * 50 + "\nPSEUDOCODE A — " + scene_a + "\n" + "=" * 50 + "\n\n")
        f.write(pseudo_a + "\n\n")
        f.write("=" * 50 + "\nPSEUDOCODE B — " + scene_b + "\n" + "=" * 50 + "\n\n")
        f.write(pseudo_b + "\n")

    print(f"\n  Saved → {cmp_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Experiment 06 — Similar-Video Comparison")
    print(f"Provider : {PROVIDER} / {MODEL}  |  Batch : {BATCH_SIZE}")
    print("=" * 60)

    results: dict[str, str] = {}  # scene_name -> pseudocode

    # Run new scenes
    for scene_name, description in SCENES.items():
        summary = run_scene(scene_name, description)
        if summary:
            results[scene_name] = summary["pseudocode"]

    # Load reuse pairs from exp05
    for scene_a, scene_b in REUSE_PAIRS:
        for scene in (scene_a, scene_b):
            if scene not in results:
                pc = load_latest_exp05_pseudocode(scene)
                if pc:
                    print(f"\n  [Reusing exp05 result for '{scene}']")
                    results[scene] = pc
                else:
                    print(f"\n  [WARNING] No exp05 result found for '{scene}'")

    # Run comparisons for all configured pairs
    comparison_pairs = [
        ("bowling_pins",    "bowling_strike"),
        ("billiard_break",  "billiards_physics"),
    ] + list(REUSE_PAIRS)

    print(f"\n\n{'=' * 60}")
    print("PAIRWISE COMPARISONS")
    print("=" * 60)

    for scene_a, scene_b in comparison_pairs:
        if scene_a in results and scene_b in results:
            run_comparison(scene_a, results[scene_a], scene_b, results[scene_b])
        else:
            missing = [s for s in (scene_a, scene_b) if s not in results]
            print(f"  SKIP pair ({scene_a}, {scene_b}) — missing: {missing}")

    print(f"\n{'=' * 60}")
    print("EXP06 COMPLETE")
    print(f"Results in {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run()
