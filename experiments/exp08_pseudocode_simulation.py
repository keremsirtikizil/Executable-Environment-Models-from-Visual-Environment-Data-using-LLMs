"""
Experiment 08 — Pseudocode → Runnable Simulation + Visualization.

Closes the full PURE loop:
  real video → pseudocode (exp05/exp06) → executable simulation → rendered output

For each selected scene the experiment:
  1. Reads the best pseudocode from exp05 / exp06 results.
  2. Asks an LLM to generate a self-contained Python simulation script
     (state machine + matplotlib renderer + GIF writer).
  3. Saves the generated code to results/exp08/<scene>/simulation.py.
  4. Executes it in a subprocess.
  5. Reports which output files were produced.

Run from project root:
  python experiments/exp08_pseudocode_simulation.py

Required env var:
  FAL_KEY = your_fal_api_key
"""

import os
import sys
import glob
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RESULTS_DIR = "results/exp08"
EXP05_DIR   = "results/exp05"
EXP06_DIR   = "results/exp06"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
PROVIDER     = "fal"
MODEL        = "google/gemini-2.5-pro"

N_STEPS      = 120   # simulation steps to render

# Scenes to simulate — (scene_name, results_dir, short description for LLM)
SCENES = [
    (
        "bouncing_ball",
        EXP05_DIR,
        "A water droplet bouncing on a superhydrophobic surface. "
        "The droplet falls under gravity, compresses on impact, then rebounds. "
        "Render as a 2D side view: surface as a horizontal line, "
        "droplet as a filled ellipse whose width/height change during compression/rebound.",
    ),
    (
        "bowling_strike",
        EXP06_DIR,
        "A bowling ball rolling down a lane and striking 10 pins arranged in a triangle. "
        "Pins fall and collide with each other. "
        "Render as a 2D top-down view: lane as a rectangle, "
        "ball as a circle, pins as small circles in the standard 4-3-2-1 triangle formation.",
    ),
    (
        "billiards_physics",
        EXP06_DIR,
        "A billiard cue ball struck and colliding with other balls on a pool table. "
        "Render as a 2D top-down view: green table, balls as colored circles, "
        "show ball trajectories / motion trails.",
    ),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_latest_pseudocode(scene_name: str, results_dir: str) -> str | None:
    pattern = os.path.join(results_dir, f"{scene_name}_*_pseudocode.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    path = matches[-1]
    print(f"  Using pseudocode: {path}")
    with open(path) as f:
        text = f.read()
    marker = "FINAL PSEUDOCODE\n" + "=" * 50
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text.strip()


def generate_simulation_code(
    scene_name: str,
    pseudocode: str,
    render_hint: str,
    output_dir: str,
) -> str:
    """Ask the LLM to produce a self-contained simulation + renderer script."""
    from openai import OpenAI

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise EnvironmentError("FAL_KEY not set.")

    client = OpenAI(
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="not-needed",
        default_headers={"Authorization": f"Key {fal_key}"},
    )

    prompt = f"""You are converting a physics pseudocode description into a runnable Python simulation.

PSEUDOCODE (extracted by PURE from a real video):
{pseudocode}

RENDERING HINT:
{render_hint}

OUTPUT REQUIREMENTS — produce ONE self-contained Python script that:

1. Implements the simulation faithfully to the pseudocode (state variables + rules).
   - Use reasonable numeric constants (e.g. gravity=0.5, restitution=0.7, etc.)
   - The simulation must actually animate — objects must move visibly across frames.

2. Renders each simulation step as a matplotlib figure and saves it as a PNG:
   - Output directory: {output_dir}
   - File names: frame_000.png, frame_001.png, …
   - Figure size: (8, 6) at 100 dpi
   - Clear, high-contrast rendering with labeled title showing step number and key state.

3. After saving all {N_STEPS} frames, creates an animated GIF:
   - Path: {output_dir}/simulation.gif
   - Frame duration: 60 ms per frame
   - Use the `Pillow` library (PIL.Image) to create the GIF from the saved PNGs.

4. Prints a one-line summary at the end: "Saved N frames + simulation.gif"

CONSTRAINTS:
- Use ONLY: matplotlib, numpy, PIL (Pillow), math, os, glob — no other dependencies.
- Do NOT use pygame, cv2, or any video library.
- The script must be importable and runnable with: python simulation.py
- Do not use plt.show() — only savefig().
- Handle edge cases (e.g. ball leaving bounds, all pins fallen) gracefully by
  resetting or freezing the simulation.

Output ONLY the Python code. No markdown fences, no explanation."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned null content.")
    # Strip markdown fences if present
    code = content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def run_simulation(code: str, scene_dir: str) -> bool:
    """Write the generated code to disk and execute it in a subprocess."""
    script_path = os.path.join(scene_dir, "simulation.py")
    with open(script_path, "w") as f:
        f.write(code)
    print(f"  Simulation code → {script_path}")
    print(f"  Running simulation ({N_STEPS} steps)...")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.stdout:
        print(f"  stdout: {result.stdout.strip()}")
    if result.stderr:
        # Print only the last few lines to avoid flooding
        lines = result.stderr.strip().split("\n")
        tail = "\n    ".join(lines[-6:])
        print(f"  stderr (last 6 lines):\n    {tail}")

    return result.returncode == 0


def count_outputs(scene_dir: str) -> dict:
    frames = sorted(glob.glob(os.path.join(scene_dir, "frame_*.png")))
    gif    = os.path.join(scene_dir, "simulation.gif")
    return {
        "n_frames": len(frames),
        "gif_exists": os.path.isfile(gif),
        "gif_path": gif if os.path.isfile(gif) else None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Experiment 08 — Pseudocode → Simulation + Visualization")
    print(f"Model  : {MODEL}  |  Steps : {N_STEPS}")
    print("=" * 60)

    all_results = []

    for scene_name, results_dir, render_hint in SCENES:
        print(f"\n{'─' * 60}")
        print(f"Scene : {scene_name}")
        print(f"{'─' * 60}")

        scene_dir = os.path.join(RESULTS_DIR, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        # Skip if already done
        existing = count_outputs(scene_dir)
        if existing["n_frames"] > 0:
            print(f"  [RESUME] {existing['n_frames']} frames already exist — skipping.")
            all_results.append({"scene": scene_name, **existing, "status": "skipped"})
            continue

        pseudocode = load_latest_pseudocode(scene_name, results_dir)
        if not pseudocode:
            print(f"  SKIP: no pseudocode found in {results_dir}")
            all_results.append({"scene": scene_name, "status": "no_pseudocode"})
            continue

        print(f"  Pseudocode length : {len(pseudocode)} chars")
        print(f"  Generating simulation code via {MODEL}...")

        try:
            code = generate_simulation_code(scene_name, pseudocode, render_hint, scene_dir)
        except Exception as e:
            print(f"  CODE GENERATION ERROR: {e}")
            all_results.append({"scene": scene_name, "status": f"codegen_error: {e}"})
            continue

        print(f"  Generated {len(code.splitlines())} lines of code.")

        try:
            success = run_simulation(code, scene_dir)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT: simulation took > 180s")
            all_results.append({"scene": scene_name, "status": "timeout"})
            continue
        except Exception as e:
            print(f"  EXECUTION ERROR: {e}")
            all_results.append({"scene": scene_name, "status": f"exec_error: {e}"})
            continue

        outputs = count_outputs(scene_dir)
        status = "ok" if success and outputs["n_frames"] > 0 else "error"
        print(f"  Status : {status} | Frames : {outputs['n_frames']} | GIF : {outputs['gif_exists']}")
        if outputs["gif_path"]:
            print(f"  GIF    → {outputs['gif_path']}")

        all_results.append({"scene": scene_name, "status": status, **outputs})

        summary_path = os.path.join(RESULTS_DIR, f"{scene_name}_{TIMESTAMP}_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"scene": scene_name, "model": MODEL, "n_steps": N_STEPS,
                       "status": status, **outputs}, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXP08 COMPLETE")
    print("=" * 60)
    for r in all_results:
        frames = r.get("n_frames", "—")
        gif    = "✓" if r.get("gif_exists") else "✗"
        print(f"  {r['scene']:25s}  frames={frames}  gif={gif}  status={r['status']}")


if __name__ == "__main__":
    run()
