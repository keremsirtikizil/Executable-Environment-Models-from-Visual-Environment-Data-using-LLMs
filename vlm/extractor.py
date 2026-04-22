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
# Incremental (multi-turn) pseudocode-only prompts
# -----------------------------------------------------------------------

INCREMENTAL_SYSTEM_PROMPT = """\
You are a careful observer. You will be shown images one batch at a time. \
After each batch, describe what you observe and what you think is happening."""

INCREMENTAL_FIRST_BATCH_PROMPT = """\
Here are {n} images in chronological order.

Look carefully at what is present and what changes between each image. \
Describe exactly what you observe, then write your best current understanding \
of the pattern or rule that governs what is happening.

Be specific — refer to what you actually see, not assumptions.

=== OBSERVATION ==="""

INCREMENTAL_NEXT_BATCH_PROMPT = """\
Here are {n} more images, continuing in the same sequence.

Study them carefully. What new things do you notice? \
Does anything confirm, contradict, or extend what you observed before?

Update your understanding. Write the complete revised description of \
what is happening and why.

=== UPDATED OBSERVATION ==="""

INCREMENTAL_FINAL_PROMPT = """\
Here are the final {n} images.

You have now seen the complete sequence. Write your final description: \
a precise, complete account of the pattern or rules governing what you observed. \
Every claim must be grounded in something you actually saw.

=== FINAL DESCRIPTION ==="""


# -----------------------------------------------------------------------
# Verification prompts — a second LLM checks the pseudocode against images
# -----------------------------------------------------------------------

VERIFY_SYSTEM_PROMPT = """\
You are a careful verifier. You will be shown a sequence of images, followed \
by a description of what someone claims is happening in them.

Your job: check every consecutive pair of images and decide whether the \
description correctly accounts for what you actually observe.

You MUST output the following sections in order:

=== FRAME-BY-FRAME ANALYSIS ===
For each pair of consecutive frames, describe:
  - What changed visually between the two frames
  - Whether the proposed description correctly predicts this change
  - If not, what the description got wrong

=== VERDICT ===
One of: CORRECT, PARTIALLY CORRECT, or INCORRECT

=== ISSUES ===
If not fully correct, list each specific claim that is wrong or missing. \
If correct, write "None"."""

VERIFY_USER_PROMPT = """\
Here is the proposed description:

{pseudocode}

Now verify this description against the image sequence above."""


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


# -----------------------------------------------------------------------
# Incremental (multi-turn) pseudocode-only extraction
# -----------------------------------------------------------------------

def extract_rule_incremental(
    image_paths: list,
    batch_size: int = 3,
    provider: str = "fal",
    model: str = "google/gemini-2.5-pro",
    verbose: bool = True,
) -> dict:
    """
    Send frames in small batches via a multi-turn conversation.
    Each round the LLM proposes or revises its pseudocode hypothesis.

    Returns dict with keys:
      pseudocode     str   final consolidated pseudocode
      rounds         list  of dicts, each with 'batch_indices', 'hypothesis', 'raw'
    """
    # Split image_paths into batches
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batches.append(image_paths[i:i + batch_size])

    if verbose:
        print(f"  Incremental extraction: {len(image_paths)} frames → "
              f"{len(batches)} batches of ~{batch_size}")

    if provider == "fal":
        return _incremental_fal(batches, model, verbose)
    elif provider == "claude":
        return _incremental_claude(batches, verbose)
    elif provider == "gpt4o":
        return _incremental_gpt4o(batches, verbose)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use: fal | claude | gpt4o")


def _build_user_message(batch, round_idx, total_batches):
    """Build one user message: images + minimal neutral prompt."""
    content = []
    for path in batch:
        content.append(_img_block(path))

    is_first = (round_idx == 0)
    is_last  = (round_idx == total_batches - 1)

    if is_first:
        prompt = INCREMENTAL_FIRST_BATCH_PROMPT.format(n=len(batch))
    elif is_last:
        prompt = INCREMENTAL_FINAL_PROMPT.format(n=len(batch))
    else:
        prompt = INCREMENTAL_NEXT_BATCH_PROMPT.format(n=len(batch))

    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _incremental_fal(batches, model, verbose):
    from openai import OpenAI

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise EnvironmentError("FAL_KEY not set.")

    client = OpenAI(
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="not-needed",
        default_headers={"Authorization": f"Key {fal_key}"},
    )

    # conversation history grows each round: [system, user, assistant, user, assistant, ...]
    conversation = [{"role": "system", "content": INCREMENTAL_SYSTEM_PROMPT}]
    rounds = []

    for r, batch in enumerate(batches):
        is_last = (r == len(batches) - 1)
        batch_start = sum(len(batches[i]) for i in range(r))
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        if verbose:
            label = "FINAL" if is_last else f"Round {r + 1}/{len(batches)}"
            print(f"  [{label}] Sending frames {batch_indices[0]}-{batch_indices[-1]}...")

        # Append next user message (images + minimal prompt, no hypothesis injected)
        conversation.append(_build_user_message(batch, r, len(batches)))

        response = client.chat.completions.create(
            model=model,
            max_tokens=16000,
            temperature=0.2,
            messages=conversation,
        )
        raw = response.choices[0].message.content.strip()

        # Append the model's actual response as the assistant turn
        conversation.append({"role": "assistant", "content": raw})

        observation = _extract_hypothesis(raw)
        rounds.append({
            "batch_indices": batch_indices,
            "hypothesis": observation,
            "raw": raw,
        })

        if verbose:
            preview = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"         → {preview}\n")

    final = rounds[-1]["hypothesis"] if rounds else ""
    return {"pseudocode": final, "rounds": rounds}


def _incremental_claude(batches, verbose):
    import anthropic
    client = anthropic.Anthropic()

    # Claude uses same alternating structure but images are base64 source blocks
    conversation = []
    rounds = []

    for r, batch in enumerate(batches):
        is_last = (r == len(batches) - 1)
        batch_start = sum(len(batches[i]) for i in range(r))
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        if verbose:
            label = "FINAL" if is_last else f"Round {r + 1}/{len(batches)}"
            print(f"  [{label}] Sending frames {batch_indices[0]}-{batch_indices[-1]}...")

        # Build user message with Claude-format image blocks
        content = []
        for path in batch:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/png", "data": _encode(path)}})
        is_first = (r == 0)
        if is_first:
            prompt = INCREMENTAL_FIRST_BATCH_PROMPT.format(n=len(batch))
        elif is_last:
            prompt = INCREMENTAL_FINAL_PROMPT.format(n=len(batch))
        else:
            prompt = INCREMENTAL_NEXT_BATCH_PROMPT.format(n=len(batch))
        content.append({"type": "text", "text": prompt})
        conversation.append({"role": "user", "content": content})

        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=16000,
            system=INCREMENTAL_SYSTEM_PROMPT,
            messages=conversation,
        )
        raw = resp.content[0].text.strip()

        # Append real assistant response to conversation
        conversation.append({"role": "assistant", "content": raw})

        observation = _extract_hypothesis(raw)
        rounds.append({"batch_indices": batch_indices, "hypothesis": observation, "raw": raw})
        if verbose:
            preview = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"         → {preview}\n")

    final = rounds[-1]["hypothesis"] if rounds else ""
    return {"pseudocode": final, "rounds": rounds}


def _incremental_gpt4o(batches, verbose):
    from openai import OpenAI
    client = OpenAI()

    conversation = [{"role": "system", "content": INCREMENTAL_SYSTEM_PROMPT}]
    rounds = []

    for r, batch in enumerate(batches):
        batch_start = sum(len(batches[i]) for i in range(r))
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        if verbose:
            is_last = (r == len(batches) - 1)
            label = "FINAL" if is_last else f"Round {r + 1}/{len(batches)}"
            print(f"  [{label}] Sending frames {batch_indices[0]}-{batch_indices[-1]}...")

        conversation.append(_build_user_message(batch, r, len(batches)))

        resp = client.chat.completions.create(
            model="gpt-4o", max_tokens=16000,
            messages=conversation,
        )
        raw = resp.choices[0].message.content.strip()
        conversation.append({"role": "assistant", "content": raw})

        observation = _extract_hypothesis(raw)
        rounds.append({"batch_indices": batch_indices, "hypothesis": observation, "raw": raw})
        if verbose:
            preview = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"         → {preview}\n")

    final = rounds[-1]["hypothesis"] if rounds else ""
    return {"pseudocode": final, "rounds": rounds}


def _extract_hypothesis(raw: str) -> str:
    """Pull the observation text from the LLM's response (after section header if present)."""
    import re
    for header in [r"=== FINAL DESCRIPTION ===",
                   r"=== UPDATED OBSERVATION ===",
                   r"=== OBSERVATION ==="]:
        match = re.search(header + r"\s*\n(.*)", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw
