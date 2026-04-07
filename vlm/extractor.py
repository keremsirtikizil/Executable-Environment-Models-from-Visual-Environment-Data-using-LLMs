"""
VLM Rule Extractor — fal.ai OpenRouter + Gemini 2.5 Flash (primary provider).

How it works:
  fal.ai hosts an OpenRouter proxy endpoint. You use the standard openai Python
  SDK pointed at fal's base_url, with FAL_KEY in the Authorization header.
  Model: "google/gemini-2.5-flash" — full vision + multimodal reasoning.

Video-like input:
  A sequence of labeled PNG frames sent in one prompt acts as a temporal
  observation stream. The model sees Frame_0 → Frame_1 → ... → Frame_N
  and reasons across time to infer transition rules. This is the same mechanism
  Gemini uses for native video (sampled frames) — so ordered PNGs IS the
  video-like interface, no video encoding needed.

Setup:
  1. Get your FAL_KEY from https://fal.ai/dashboard
  2. In PyCharm: Run > Edit Configurations > Environment Variables
       FAL_KEY = your_key_here
     OR create a .env file in project root:
       FAL_KEY=your_key_here
     Then: pip install python-dotenv
     And add at top of your script: from dotenv import load_dotenv; load_dotenv()
"""

import os
import base64


SYSTEM_PROMPT = """\
You are an expert program synthesizer. You will be shown a sequence of images \
depicting consecutive states of an UNKNOWN 2D grid environment. The rules are \
completely invented — you have never seen them before. Do NOT assume this \
resembles any known game.

Your ONLY input is the images. Study them carefully and infer ALL rules that \
govern how the environment transitions from one state to the next.

You MUST output TWO sections, in this exact order:

=== PSEUDOCODE ===
Write a high-level, human-readable pseudocode description of the transition rules \
you inferred. Use plain English with simple indentation. No Python syntax — just \
logic a human can skim in 10 seconds. Keep it short and abstract — capture the \
RULES, not implementation details.

=== PYTHON ===
Output ONLY a valid Python function — no markdown fences, no explanation outside \
the function body.

Required signature:
def apply_action(grid, agent_pos, action):
    \"\"\"
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    \"\"\"
    ...
"""

SEQUENCE_PROMPT = """\
These {n} images are consecutive frames in chronological order. \
Study every detail that changes between frames and infer the complete rules."""

PAIR_PROMPT = """\
The first image is BEFORE and the second image is AFTER a single action. \
Study what changed and infer the transition rule."""


# -----------------------------------------------------------------------
# Verification prompts — a second LLM checks the pseudocode against images
# -----------------------------------------------------------------------

VERIFY_SYSTEM_PROMPT = """\
You are a careful verifier. You will be shown a sequence of images depicting \
consecutive states of a 2D grid environment, followed by a proposed set of rules \
(pseudocode) that someone claims describe how the environment works.

Your job: compare the proposed rules against what you actually observe in the \
images. Check EVERY transition between consecutive frames.

You MUST output the following sections in order:

=== FRAME-BY-FRAME ANALYSIS ===
For each pair of consecutive frames, describe:
  - What changed visually between the two frames
  - Whether the proposed rules correctly predict this change
  - If not, what the rules got wrong

=== VERDICT ===
One of: CORRECT, PARTIALLY CORRECT, or INCORRECT

=== ISSUES ===
If not fully correct, list each specific rule that is wrong or missing. \
If correct, write "None"."""

VERIFY_USER_PROMPT = """\
Here are the proposed rules (pseudocode):

{pseudocode}

Now verify these rules against the image sequence above."""


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _img_block(path: str) -> dict:
    """OpenAI-format image_url content block from a local PNG."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{_encode(path)}",
            "detail": "high"
        }
    }


# -----------------------------------------------------------------------
# PRIMARY: fal.ai OpenRouter — Gemini 2.5 Flash
# -----------------------------------------------------------------------

def extract_rule_fal(
    image_paths: list,
    action_name: str = "",
    model: str = "google/gemini-2.5-flash",
    is_sequence: bool = False,
    action_sequence: list = None
) -> str:
    """
    Send frames to Gemini 2.5 Flash via fal.ai OpenRouter and get back
    a Python transition function.

    image_paths:     PNG paths in temporal order (2 for pair, N for sequence)
    action_name:     human label used in pair mode
    model:           any OpenRouter vision model string
    is_sequence:     True = video-like multi-frame mode
    action_sequence: action labels between frames (len = len(image_paths) - 1)
    """
    from openai import OpenAI

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise EnvironmentError(
            "FAL_KEY not set. Get it at https://fal.ai/dashboard\n"
            "PyCharm: Run > Edit Configurations > Environment Variables > FAL_KEY=..."
        )

    client = OpenAI(
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="not-needed",
        default_headers={"Authorization": f"Key {fal_key}"},
    )

    content = []

    if is_sequence and action_sequence:
        # Video-like: images only, no action labels
        for i, path in enumerate(image_paths):
            content.append(_img_block(path))
        content.append({"type": "text", "text": SEQUENCE_PROMPT.format(
            n=len(image_paths)
        )})
    else:
        for path in image_paths:
            content.append(_img_block(path))
        content.append({"type": "text", "text": PAIR_PROMPT})

    response = client.chat.completions.create(
        model=model,
        max_tokens=32000,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
    )
    raw = response.choices[0].message.content.strip()
    return _split_pseudocode_and_python(raw)


def _strip_fences(text: str) -> str:
    """Extract code from markdown fences; fall back to raw text if none found."""
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _split_pseudocode_and_python(raw: str) -> dict:
    """
    Parse the LLM response into pseudocode and python sections.
    Returns dict with keys 'pseudocode' and 'python'.
    """
    import re
    pseudo = ""
    python = ""

    # Try splitting on section headers
    pseudo_match = re.search(
        r"=== PSEUDOCODE ===\s*\n(.*?)(?=\n=== PYTHON ===)", raw, re.DOTALL
    )
    python_match = re.search(
        r"=== PYTHON ===\s*\n(.*)", raw, re.DOTALL
    )

    if pseudo_match:
        pseudo = pseudo_match.group(1).strip()
    if python_match:
        python = _strip_fences(python_match.group(1).strip())
    else:
        # Fallback: treat entire output as python (backward compat)
        python = _strip_fences(raw)

    return {"pseudocode": pseudo, "python": python}


# -----------------------------------------------------------------------
# Pseudocode verification — second LLM checks rules against images
# -----------------------------------------------------------------------

def verify_pseudocode(
    image_paths: list,
    pseudocode: str,
    provider: str = "fal",
    model: str = "google/gemini-2.5-flash",
) -> dict:
    """
    Send the same images + the generated pseudocode to a (different) LLM.
    It checks whether the pseudocode correctly describes the observed transitions.

    Returns dict with keys:
      raw           str   full verifier response
      verdict       str   CORRECT / PARTIALLY CORRECT / INCORRECT
      issues        str   list of issues (or "None")
      frame_analysis str  per-frame breakdown
    """
    if provider == "fal":
        return _verify_fal(image_paths, pseudocode, model)
    elif provider == "claude":
        return _verify_claude(image_paths, pseudocode)
    elif provider == "gpt4o":
        return _verify_gpt4o(image_paths, pseudocode)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use: fal | claude | gpt4o")


def _verify_fal(image_paths, pseudocode, model):
    from openai import OpenAI

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise EnvironmentError("FAL_KEY not set.")

    client = OpenAI(
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="not-needed",
        default_headers={"Authorization": f"Key {fal_key}"},
    )

    content = []
    for path in image_paths:
        content.append(_img_block(path))
    content.append({"type": "text", "text": VERIFY_USER_PROMPT.format(
        pseudocode=pseudocode)})

    response = client.chat.completions.create(
        model=model,
        max_tokens=16000,
        temperature=0.2,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
    )
    raw = response.choices[0].message.content.strip()
    return _parse_verification(raw)


def _verify_claude(image_paths, pseudocode):
    import anthropic
    client = anthropic.Anthropic()
    content = []
    for p in image_paths:
        content.append({"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": _encode(p)}})
    content.append({"type": "text", "text": VERIFY_USER_PROMPT.format(
        pseudocode=pseudocode)})
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=8000,
                               system=VERIFY_SYSTEM_PROMPT,
                               messages=[{"role": "user", "content": content}])
    return _parse_verification(r.content[0].text.strip())


def _verify_gpt4o(image_paths, pseudocode):
    from openai import OpenAI
    client = OpenAI()
    content = []
    for p in image_paths:
        content.append(_img_block(p))
    content.append({"type": "text", "text": VERIFY_USER_PROMPT.format(
        pseudocode=pseudocode)})
    r = client.chat.completions.create(
        model="gpt-4o", max_tokens=8000,
        messages=[{"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                  {"role": "user", "content": content}])
    return _parse_verification(r.choices[0].message.content.strip())


def _parse_verification(raw: str) -> dict:
    """Parse the verifier response into structured fields."""
    import re

    verdict = "UNKNOWN"
    issues = ""
    frame_analysis = ""

    v_match = re.search(r"=== VERDICT ===\s*\n(.*?)(?=\n===|\Z)", raw, re.DOTALL)
    if v_match:
        verdict = v_match.group(1).strip().split("\n")[0].strip()

    i_match = re.search(r"=== ISSUES ===\s*\n(.*?)(?=\n===|\Z)", raw, re.DOTALL)
    if i_match:
        issues = i_match.group(1).strip()

    f_match = re.search(
        r"=== FRAME-BY-FRAME ANALYSIS ===\s*\n(.*?)(?=\n=== VERDICT ===|\Z)",
        raw, re.DOTALL
    )
    if f_match:
        frame_analysis = f_match.group(1).strip()

    return {
        "raw": raw,
        "verdict": verdict,
        "issues": issues,
        "frame_analysis": frame_analysis,
    }


# -----------------------------------------------------------------------
# GIF recorder
# -----------------------------------------------------------------------

def record_episode_gif(env, actions, output_path, frame_duration_ms=600, reset_grid=None,
                       label_frames=False):
    from magnet_env.magnet_world import ACTION_NAMES
    if reset_grid is not None:
        env.reset(reset_grid)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if label_frames:
        frames = [env.render_with_label("Initial state").convert("RGB")]
    else:
        frames = [env.render_with_label("").convert("RGB")]
    for i, action in enumerate(actions):
        env.step(action)
        if label_frames:
            lbl = f"Step {i+1}: {ACTION_NAMES.get(action, action)}"
        else:
            lbl = ""
        frames.append(env.render_with_label(lbl).convert("RGB"))
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=frame_duration_ms, loop=0, optimize=False)
    print(f"  GIF → {output_path}  ({len(frames)} frames)")
    return frames


# -----------------------------------------------------------------------
# Save episode as individual PNGs (feed into extract_rule_fal)
# -----------------------------------------------------------------------

def save_frames_as_images(env, actions, output_dir, prefix="frame", reset_grid=None,
                          label_frames=False):
    """Returns list of PNG paths in temporal order."""
    from magnet_env.magnet_world import ACTION_NAMES
    if reset_grid is not None:
        env.reset(reset_grid)
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    p = os.path.join(output_dir, f"{prefix}_step0.png")
    lbl = "Initial state" if label_frames else ""
    env.render_with_label(lbl).save(p)
    paths.append(p)
    for i, action in enumerate(actions):
        env.step(action)
        p = os.path.join(output_dir, f"{prefix}_step{i+1}.png")
        lbl = f"Step {i+1}: {ACTION_NAMES.get(action, action)}" if label_frames else ""
        env.render_with_label(lbl).save(p)
        paths.append(p)
    return paths


# -----------------------------------------------------------------------
# Unified entry point
# -----------------------------------------------------------------------

def extract_rule(
    image_paths: list,
    action_name: str = "",
    provider: str = "fal",
    model: str = "google/gemini-2.5-flash",
    is_sequence: bool = False,
    action_sequence: list = None
) -> str:
    """
    provider: "fal"    — fal.ai OpenRouter, Gemini 2.5 Flash (needs FAL_KEY)
              "claude" — direct Anthropic API   (needs ANTHROPIC_API_KEY)
              "gpt4o"  — direct OpenAI API      (needs OPENAI_API_KEY)

    Other fal.ai models you can try (just change model= string):
      "google/gemini-2.5-pro"            stronger reasoning, costs more
      "anthropic/claude-sonnet-4-6"      Claude via OpenRouter on fal
      "openai/gpt-4o"                    GPT-4o via OpenRouter on fal
    """
    if provider == "fal":
        return extract_rule_fal(image_paths, action_name, model=model,
                                is_sequence=is_sequence, action_sequence=action_sequence)
    elif provider == "claude":
        return _claude(image_paths, action_name, is_sequence, action_sequence)
    elif provider == "gpt4o":
        return _gpt4o(image_paths, action_name, is_sequence, action_sequence)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use: fal | claude | gpt4o")


def _claude(image_paths, action_name, is_sequence, action_sequence):
    import anthropic
    client = anthropic.Anthropic()
    content = []
    if is_sequence and action_sequence:
        for p in image_paths:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/png", "data": _encode(p)}})
        content.append({"type": "text", "text": SEQUENCE_PROMPT.format(
            n=len(image_paths))})
    else:
        for p in image_paths:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/png", "data": _encode(p)}})
        content.append({"type": "text", "text": PAIR_PROMPT})
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=4000,
                               system=SYSTEM_PROMPT,
                               messages=[{"role": "user", "content": content}])
    return _split_pseudocode_and_python(r.content[0].text.strip())


def _gpt4o(image_paths, action_name, is_sequence, action_sequence):
    from openai import OpenAI
    client = OpenAI()
    content = []
    if is_sequence and action_sequence:
        for p in image_paths:
            content.append(_img_block(p))
        content.append({"type": "text", "text": SEQUENCE_PROMPT.format(
            n=len(image_paths))})
    else:
        for p in image_paths:
            content.append(_img_block(p))
        content.append({"type": "text", "text": PAIR_PROMPT})
    r = client.chat.completions.create(
        model="gpt-4o", max_tokens=4000,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}])
    return _split_pseudocode_and_python(r.choices[0].message.content.strip())
