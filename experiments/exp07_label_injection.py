"""
Experiment 07 — Label Injection Bias Test.

Tests whether injecting a natural-language scene description into the system
prompt biases PURE's pseudocode output.

Research question: Is the model reading the frames, or is it just elaborating
on the label?

Design:
  For each of 2 scenes (bowling_strike, billiards_physics) run PURE 3 times:
    no_label      — no description injected (control, same as exp05/exp06)
    correct_label — true description injected into the system prompt
    wrong_label   — completely wrong description injected (cross-matched)

  After all 6 runs, use the VLM to compare the 3 pseudocodes per scene and
  assess how much the label changed what physics was extracted.

Frames used:
  frames/realworld/bowling_strike/    (31 frames, from exp06)
  frames/realworld/billiards_physics/ (36 frames, from exp06)

Run from project root:
  python experiments/exp07_label_injection.py

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

from vlm.extractor import extract_rule_incremental

FRAMES_ROOT = "frames/realworld"
RESULTS_DIR = "results/exp07"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
PROVIDER     = "fal"
MODEL        = "google/gemini-2.5-pro"
VERIFY_MODEL = "google/gemini-2.5-flash"
BATCH_SIZE   = 3

# Scene configs: each scene maps condition -> hint string (None = no hint)
SCENES = {
    "bowling_strike": {
        "correct": "A bowling ball rolling down a lane and striking all ten pins — a perfect strike.",
        "wrong":   "A pendulum swinging back and forth in slow motion under gravity.",
    },
    "billiards_physics": {
        "correct": "Billiard balls rolling and colliding on a pool table — various shots demonstrating physics.",
        "wrong":   "A water droplet bouncing on a superhydrophobic surface in slow motion.",
    },
}

CONDITIONS = ["no_label", "correct_label", "wrong_label"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_frames(scene_name: str) -> list:
    d = os.path.join(FRAMES_ROOT, scene_name)
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames in {d}")
    return paths


def result_path(scene: str, condition: str, kind: str) -> str:
    """Return the canonical path for a result file (pseudocode/rounds/summary)."""
    return os.path.join(RESULTS_DIR, f"{scene}_{condition}_{TIMESTAMP}_{kind}.txt")


def existing_pseudocode(scene: str, condition: str) -> str | None:
    """Return pseudocode text if a result already exists for this scene+condition."""
    pattern = os.path.join(RESULTS_DIR, f"{scene}_{condition}_*_pseudocode.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        text = f.read()
    marker = "FINAL PSEUDOCODE\n" + "=" * 50
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text.strip()


# ── single run ────────────────────────────────────────────────────────────────

def run_condition(scene: str, condition: str, hint: str | None) -> str | None:
    """
    Run PURE on `scene` with the given `hint` (or None for no_label).
    Returns the final pseudocode string, or None on error.
    Skips and returns cached pseudocode if the condition was already run.
    """
    cached = existing_pseudocode(scene, condition)
    if cached is not None:
        print(f"\n  [RESUME] Skipping {scene}/{condition} — result already exists.")
        return cached

    hint_display = f'"{hint}"' if hint else "None"
    print(f"\n{'=' * 60}")
    print(f"Scene     : {scene}")
    print(f"Condition : {condition}")
    print(f"Hint      : {hint_display}")
    print(f"{'=' * 60}")

    try:
        frame_paths = load_frames(scene)
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
            scene_hint=hint,
        )
    except Exception as e:
        print(f"  EXTRACTION ERROR: {e}")
        import traceback; traceback.print_exc()
        return None

    pseudocode = result["pseudocode"]
    rounds     = result["rounds"]

    # pseudocode file
    pseudo_path = result_path(scene, condition, "pseudocode")
    with open(pseudo_path, "w") as f:
        f.write(f"Scene     : {scene}\n")
        f.write(f"Condition : {condition}\n")
        f.write(f"Hint      : {hint_display}\n")
        f.write(f"Provider  : {PROVIDER} / {MODEL}\n")
        f.write(f"Frames    : {len(frame_paths)} | Batch : {BATCH_SIZE} | Rounds : {len(rounds)}\n\n")
        f.write("=" * 50 + "\nFINAL PSEUDOCODE\n" + "=" * 50 + "\n\n")
        f.write(pseudocode)

    # rounds file
    rounds_path = result_path(scene, condition, "rounds")
    with open(rounds_path, "w") as f:
        for i, rnd in enumerate(rounds):
            label = "FINAL" if i == len(rounds) - 1 else f"Round {i + 1}"
            f.write(f"\n{'=' * 50}\n{label}  —  frames {rnd['batch_indices']}\n{'=' * 50}\n\n")
            f.write(rnd["raw"] + "\n")

    # summary JSON
    summary = {
        "experiment" : "exp07_label_injection",
        "scene"      : scene,
        "condition"  : condition,
        "hint"       : hint,
        "provider"   : PROVIDER,
        "model"      : MODEL,
        "batch_size" : BATCH_SIZE,
        "n_frames"   : len(frame_paths),
        "n_rounds"   : len(rounds),
        "pseudocode" : pseudocode,
    }
    summary_path = os.path.join(
        RESULTS_DIR, f"{scene}_{condition}_{TIMESTAMP}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "pseudocode"}, f, indent=2)

    print(f"\n--- FINAL PSEUDOCODE ---\n{pseudocode}\n{'- ' * 20}")
    print(f"  Pseudocode → {pseudo_path}")

    return pseudocode


# ── VLM label-injection analysis ──────────────────────────────────────────────

def compare_conditions(
    scene: str,
    pseudo_no_label: str,
    pseudo_correct: str,
    pseudo_wrong: str,
) -> str:
    """
    Ask the VLM to compare the 3 pseudocodes produced under different label
    conditions and assess whether label injection changed the extracted physics.
    """
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
        f"You are analysing PURE, a physics-rule extraction system.\n\n"
        f"PURE was run three times on the same video ({scene}) with different "
        f"system-prompt conditions:\n\n"
        f"  no_label      — no scene description provided (control)\n"
        f"  correct_label — the true scene description was injected\n"
        f"  wrong_label   — a completely unrelated scene description was injected\n\n"
        f"The three resulting pseudocodes are shown below.\n\n"
        f"{'=' * 40}\n"
        f"NO_LABEL (control)\n"
        f"{'=' * 40}\n"
        f"{pseudo_no_label}\n\n"
        f"{'=' * 40}\n"
        f"CORRECT_LABEL\n"
        f"{'=' * 40}\n"
        f"{pseudo_correct}\n\n"
        f"{'=' * 40}\n"
        f"WRONG_LABEL\n"
        f"{'=' * 40}\n"
        f"{pseudo_wrong}\n\n"
        f"Please answer the following questions concisely, using bullet points:\n\n"
        f"1. CORRECT vs NO_LABEL — what changed in the extracted physics when the "
        f"true scene description was provided?\n"
        f"2. WRONG vs NO_LABEL — what changed (if anything) when a false scene "
        f"description was injected? Did the model start hallucinating the wrong "
        f"physics (e.g. pendulum motion instead of bowling)?\n"
        f"3. SENSITIVITY — Rate overall label sensitivity: HIGH / MEDIUM / LOW. "
        f"HIGH = the label substantially changed the physics extracted; "
        f"MEDIUM = minor wording differences only; "
        f"LOW = pseudocodes are essentially identical across all three conditions. "
        f"Justify your rating.\n"
        f"4. CONCLUSION — Is PURE primarily reading the frames, or primarily "
        f"elaborating on the label? Support with evidence from the pseudocodes."
    )

    response = client.chat.completions.create(
        model=VERIFY_MODEL,
        max_tokens=4000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def run_analysis(scene: str, pseudocodes: dict[str, str]) -> None:
    """Run and save the 3-way comparison analysis for a scene."""
    # Skip if analysis already exists
    existing = sorted(glob.glob(
        os.path.join(RESULTS_DIR, f"analysis_{scene}_*.txt")
    ))
    if existing:
        print(f"\n  [RESUME] Skipping analysis for '{scene}' — already exists.")
        return

    print(f"\n{'#' * 60}")
    print(f"LABEL INJECTION ANALYSIS — {scene}")
    print(f"{'#' * 60}")

    try:
        analysis = compare_conditions(
            scene=scene,
            pseudo_no_label=pseudocodes["no_label"],
            pseudo_correct=pseudocodes["correct_label"],
            pseudo_wrong=pseudocodes["wrong_label"],
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        analysis = f"Analysis error: {e}"

    print(analysis)

    analysis_path = os.path.join(RESULTS_DIR, f"analysis_{scene}_{TIMESTAMP}.txt")
    with open(analysis_path, "w") as f:
        f.write(f"Scene    : {scene}\n")
        f.write(f"Model    : {PROVIDER} / {VERIFY_MODEL}\n\n")
        f.write("=" * 50 + "\nLABEL INJECTION ANALYSIS\n" + "=" * 50 + "\n\n")
        f.write(analysis + "\n\n")
        f.write("=" * 50 + "\nPSEUDOCODE — no_label\n" + "=" * 50 + "\n\n")
        f.write(pseudocodes["no_label"] + "\n\n")
        f.write("=" * 50 + "\nPSEUDOCODE — correct_label\n" + "=" * 50 + "\n\n")
        f.write(pseudocodes["correct_label"] + "\n\n")
        f.write("=" * 50 + "\nPSEUDOCODE — wrong_label\n" + "=" * 50 + "\n\n")
        f.write(pseudocodes["wrong_label"] + "\n")

    print(f"\n  Saved → {analysis_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Experiment 07 — Label Injection Bias Test")
    print(f"Provider : {PROVIDER} / {MODEL}  |  Batch : {BATCH_SIZE}")
    print("=" * 60)

    # Collect pseudocodes: scene -> condition -> pseudocode_str
    all_pseudocodes: dict[str, dict[str, str]] = {s: {} for s in SCENES}

    for scene, hints in SCENES.items():
        condition_hints = {
            "no_label"      : None,
            "correct_label" : hints["correct"],
            "wrong_label"   : hints["wrong"],
        }

        for condition in CONDITIONS:
            hint = condition_hints[condition]
            pseudocode = run_condition(scene, condition, hint)
            if pseudocode is not None:
                all_pseudocodes[scene][condition] = pseudocode
            else:
                print(f"  WARNING: no pseudocode for {scene}/{condition} — skipping analysis.")

    # Run per-scene 3-way analyses
    print(f"\n\n{'=' * 60}")
    print("LABEL INJECTION ANALYSES")
    print("=" * 60)

    for scene in SCENES:
        pcs = all_pseudocodes[scene]
        if len(pcs) == 3:
            run_analysis(scene, pcs)
        else:
            missing = [c for c in CONDITIONS if c not in pcs]
            print(f"\n  SKIP analysis for '{scene}' — missing conditions: {missing}")

    print(f"\n{'=' * 60}")
    print("EXP07 COMPLETE")
    print(f"Results in {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run()
