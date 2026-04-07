"""
Generate a publication-quality pipeline diagram for PURE.
Outputs: results/pipeline_diagram.png
"""

from PIL import Image, ImageDraw, ImageFont
import os


# ── Canvas ──────────────────────────────────────────────────────────────
W, H = 1400, 900
BG = (255, 255, 255)
PAD = 50

# ── Color palette ───────────────────────────────────────────────────────
C_ENV_FILL    = (234, 241, 252)
C_ENV_STROKE  = (60, 90, 155)
C_LLM_FILL    = (255, 241, 218)
C_LLM_STROKE  = (190, 120, 30)
C_EVAL_FILL   = (224, 245, 224)
C_EVAL_STROKE = (50, 130, 60)
C_OUT_FILL    = (242, 234, 250)
C_OUT_STROKE  = (115, 65, 160)
C_ARROW       = (70, 70, 70)
C_TEXT        = (25, 25, 25)
C_SUB         = (90, 90, 90)
C_RED         = (175, 45, 45)
C_TITLE       = (25, 40, 95)
C_DIV         = (195, 195, 195)
C_BADGE_BG    = (60, 90, 155)
C_BADGE_TEXT  = (255, 255, 255)


# ── Font loader ─────────────────────────────────────────────────────────
def _font(size):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

F_TITLE   = _font(24)
F_SUBTITLE = _font(13)
F_HEAD    = _font(14)
F_BODY    = _font(12)
F_SMALL   = _font(11)
F_BADGE   = _font(11)
F_FILE    = _font(10)


# ── Drawing helpers ─────────────────────────────────────────────────────
def box(draw, x, y, w, h, fill, stroke, radius=12, lw=2):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius,
                           fill=fill, outline=stroke, width=lw)


def badge(draw, text, x, y):
    """Numbered step badge — drawn ABOVE the box top-left."""
    tw = draw.textbbox((0, 0), text, font=F_BADGE)[2] + 14
    bh = 20
    draw.rounded_rectangle([x, y - bh - 4, x + tw, y - 4],
                           radius=10, fill=C_BADGE_BG)
    draw.text((x + 7, y - bh - 2), text, fill=C_BADGE_TEXT, font=F_BADGE)


def arrow_r(draw, x0, y, x1):
    draw.line([(x0, y), (x1 - 8, y)], fill=C_ARROW, width=2)
    draw.polygon([(x1, y), (x1 - 10, y - 5), (x1 - 10, y + 5)], fill=C_ARROW)


def arrow_d(draw, x, y0, y1):
    draw.line([(x, y0), (x, y1 - 8)], fill=C_ARROW, width=2)
    draw.polygon([(x, y1), (x - 5, y1 - 10), (x + 5, y1 - 10)], fill=C_ARROW)


def txt(draw, text, x, y, font=F_BODY, fill=C_TEXT):
    draw.text((x, y), text, fill=fill, font=font)


def txt_c(draw, text, cx, y, font=F_BODY, fill=C_TEXT):
    bb = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (bb[2] - bb[0]) // 2, y), text, fill=fill, font=font)


# ── Main ────────────────────────────────────────────────────────────────
def generate():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Title
    txt_c(d, "PURE \u2014 Pipeline Overview", W // 2, 20, F_TITLE, C_TITLE)
    txt_c(d, "Program Undergrad Research \u00b7 Sabanci University",
          W // 2, 50, F_SUBTITLE, C_SUB)
    d.line([(PAD, 76), (W - PAD, 76)], fill=C_DIV, width=1)

    # ── ROW 1 ───────────────────────────────────────────────────────
    r1y = 100
    bh1 = 120
    bw = 340
    gap = 50

    # Box 1 — Environment
    x1 = PAD
    badge(d, "Step 1", x1, r1y)
    box(d, x1, r1y, bw, bh1, C_ENV_FILL, C_ENV_STROKE)
    txt(d, "MagnetWorld Environment", x1 + 16, r1y + 12, F_HEAD, C_ENV_STROKE)
    txt(d, "Invented 2D grid \u00b7 5 hidden rules", x1 + 16, r1y + 34, F_BODY, C_SUB)
    txt(d, "6 cell types: empty, wall, metal,", x1 + 16, r1y + 56, F_SMALL, C_SUB)
    txt(d, "ice, hole, agent", x1 + 16, r1y + 72, F_SMALL, C_SUB)
    txt(d, "Scripted episodes (action sequences)", x1 + 16, r1y + 94, F_SMALL, C_SUB)

    # Arrow
    a1 = x1 + bw
    a2 = a1 + gap
    arrow_r(d, a1, r1y + bh1 // 2, a2)
    txt_c(d, "step()", (a1 + a2) // 2, r1y + bh1 // 2 - 18, F_SMALL, C_SUB)

    # Box 2 — Frame Generation
    x2 = a2
    badge(d, "Step 2", x2, r1y)
    box(d, x2, r1y, bw, bh1, C_ENV_FILL, C_ENV_STROKE)
    txt(d, "Frame Generation", x2 + 16, r1y + 12, F_HEAD, C_ENV_STROKE)
    txt(d, "Render each grid state as PNG", x2 + 16, r1y + 34, F_BODY, C_SUB)
    txt(d, "NO text labels on frames", x2 + 16, r1y + 56, F_SMALL, C_RED)
    txt(d, "NO action names provided", x2 + 16, r1y + 72, F_SMALL, C_RED)
    txt(d, "Also saves GIF for human review", x2 + 16, r1y + 94, F_SMALL, C_SUB)

    # Arrow
    a3 = x2 + bw
    a4 = a3 + gap
    arrow_r(d, a3, r1y + bh1 // 2, a4)
    txt_c(d, "images only", (a3 + a4) // 2, r1y + bh1 // 2 - 18, F_SMALL, C_RED)

    # Box 3 — LLM Extraction
    x3 = a4
    bw3 = W - PAD - x3
    badge(d, "Step 3 \u2014 LLM Call #1", x3, r1y)
    box(d, x3, r1y, bw3, bh1, C_LLM_FILL, C_LLM_STROKE)
    txt(d, "Rule Extraction (VLM)", x3 + 16, r1y + 12, F_HEAD, C_LLM_STROKE)
    txt(d, "Sees ONLY the frame images", x3 + 16, r1y + 34, F_BODY, C_SUB)
    txt(d, "Gemini 2.5 Pro / Claude / GPT-4o", x3 + 16, r1y + 56, F_SMALL, C_SUB)
    txt(d, "Outputs:", x3 + 16, r1y + 78, F_BODY, C_LLM_STROKE)
    txt(d, "  Pseudocode  +  Python apply_action()", x3 + 16, r1y + 96, F_SMALL, C_SUB)

    # ── Vertical arrows from Box 3 ──────────────────────────────────
    r2y = r1y + bh1 + 80
    bh2 = 145

    # Left split: pseudocode → verification
    left_cx = PAD + (W // 2 - PAD) // 2
    right_cx = W // 2 + (W - PAD - W // 2) // 2

    # Draw the split
    mid_bot = r1y + bh1
    mid_y = mid_bot + 32

    # Vertical from box 3 center
    box3_cx = x3 + bw3 // 2
    arrow_d(d, box3_cx, mid_bot, mid_bot + 18)

    # Horizontal bar
    d.line([(left_cx, mid_y), (right_cx, mid_y)], fill=C_ARROW, width=2)
    d.line([(box3_cx, mid_bot + 18), (box3_cx, mid_y)], fill=C_ARROW, width=2)
    d.ellipse([box3_cx - 4, mid_y - 4, box3_cx + 4, mid_y + 4], fill=C_ARROW)

    # Down to left box
    arrow_d(d, left_cx, mid_y, r2y)
    txt(d, "pseudocode", left_cx + 8, mid_y + 4, F_SMALL, C_SUB)

    # Down to right box
    arrow_d(d, right_cx, mid_y, r2y)
    txt(d, "Python code", right_cx + 8, mid_y + 4, F_SMALL, C_SUB)

    # ── ROW 2 ───────────────────────────────────────────────────────
    bw2 = (W - PAD * 2 - 50) // 2

    # Box 4 — Verification
    x4 = PAD
    badge(d, "Step 4 \u2014 LLM Call #2", x4, r2y)
    box(d, x4, r2y, bw2, bh2, C_LLM_FILL, C_LLM_STROKE)
    txt(d, "Pseudocode Verification", x4 + 16, r2y + 12, F_HEAD, C_LLM_STROKE)
    txt(d, "Different model verifies the pseudocode", x4 + 16, r2y + 34, F_BODY, C_SUB)
    txt(d, "Input: same images + generated pseudocode", x4 + 16, r2y + 56, F_SMALL, C_SUB)
    txt(d, "For each consecutive frame pair:", x4 + 16, r2y + 76, F_SMALL, C_SUB)
    txt(d, "  \u201cDo the proposed rules predict this change?\u201d", x4 + 16, r2y + 92, F_SMALL, C_SUB)
    txt(d, "Verdict:", x4 + 16, r2y + 116, F_BODY, C_LLM_STROKE)
    txt(d, "CORRECT / PARTIALLY CORRECT / INCORRECT", x4 + 80, r2y + 116, F_BODY, C_LLM_STROKE)

    # Box 5 — Evaluation
    x5 = PAD + bw2 + 50
    badge(d, "Step 5 \u2014 Ground Truth", x5, r2y)
    box(d, x5, r2y, bw2, bh2, C_EVAL_FILL, C_EVAL_STROKE)
    txt(d, "Code Evaluation", x5 + 16, r2y + 12, F_HEAD, C_EVAL_STROKE)
    txt(d, "Execute apply_action() against", x5 + 16, r2y + 34, F_BODY, C_SUB)
    txt(d, "31 hand-crafted test cases", x5 + 16, r2y + 52, F_BODY, C_EVAL_STROKE)
    txt(d, "Covers all 5 rules \u00d7 all directions \u00d7 edge cases", x5 + 16, r2y + 74, F_SMALL, C_SUB)
    txt(d, "Compare agent_pos & metal_pos", x5 + 16, r2y + 94, F_SMALL, C_SUB)
    txt(d, "vs. expected ground-truth positions", x5 + 16, r2y + 110, F_SMALL, C_SUB)
    txt(d, "Output: accuracy (passed / total)", x5 + 16, r2y + 130, F_BODY, C_EVAL_STROKE)

    # ── ROW 3 — Output ─────────────────────────────────────────────
    r3y = r2y + bh2 + 65
    bh3 = 108

    # Arrows converging
    arrow_d(d, left_cx, r2y + bh2, r2y + bh2 + 22)
    arrow_d(d, right_cx, r2y + bh2, r2y + bh2 + 22)
    merge_y = r2y + bh2 + 22
    d.line([(left_cx, merge_y), (right_cx, merge_y)], fill=C_ARROW, width=2)
    center_x = W // 2
    d.ellipse([center_x - 4, merge_y - 4, center_x + 4, merge_y + 4], fill=C_ARROW)
    arrow_d(d, center_x, merge_y, r3y)

    # Box 6 — Visualization
    bw6 = W - PAD * 2
    x6 = PAD
    badge(d, "Step 6 \u2014 Results", x6, r3y)
    box(d, x6, r3y, bw6, bh3, C_OUT_FILL, C_OUT_STROKE)
    txt(d, "Visualization & Output", x6 + 16, r3y + 12, F_HEAD, C_OUT_STROKE)
    txt(d, "Visual report: INPUT | EXPECTED | GOT  for every test case "
        "(all 6 cell types rendered, green/red borders)", x6 + 16, r3y + 36, F_BODY, C_SUB)

    # File list
    files = [
        ("*_extracted.py", "Python function"),
        ("*_pseudocode.txt", "Inferred rules"),
        ("*_verification.txt", "Verifier verdict"),
        ("*_visual.png", "Grid comparison"),
        ("*_summary.json", "Metrics & paths"),
    ]
    fx = x6 + 24
    fy = r3y + 64
    for fname, desc in files:
        # File icon (small box)
        d.rounded_rectangle([fx, fy, fx + 12, fy + 14], radius=2,
                            fill=C_OUT_FILL, outline=C_OUT_STROKE, width=1)
        d.line([(fx + 2, fy + 4), (fx + 10, fy + 4)], fill=C_OUT_STROKE, width=1)
        d.line([(fx + 2, fy + 8), (fx + 8, fy + 8)], fill=C_OUT_STROKE, width=1)
        txt(d, fname, fx + 18, fy - 1, F_FILE, C_OUT_STROKE)
        txt(d, desc, fx + 18, fy + 13, F_FILE, C_SUB)
        fx += 250

    # ── Legend ──────────────────────────────────────────────────────
    ly = r3y + bh3 + 22
    d.line([(PAD, ly - 6), (W - PAD, ly - 6)], fill=C_DIV, width=1)

    items = [
        (C_ENV_FILL, C_ENV_STROKE, "Environment / Processing"),
        (C_LLM_FILL, C_LLM_STROKE, "LLM Calls (vision models)"),
        (C_EVAL_FILL, C_EVAL_STROKE, "Deterministic Evaluation"),
        (C_OUT_FILL, C_OUT_STROKE, "Output Artifacts"),
    ]
    lx = PAD + 30
    for fill, stroke, label in items:
        d.rounded_rectangle([lx, ly, lx + 20, ly + 14], radius=4,
                            fill=fill, outline=stroke, width=1)
        txt(d, label, lx + 28, ly, F_SMALL, C_TEXT)
        lx += 280

    # Save
    os.makedirs("results", exist_ok=True)
    out = "results/pipeline_diagram.png"
    img.save(out, dpi=(300, 300))
    print(f"Pipeline diagram saved \u2192 {out}")
    return out


if __name__ == "__main__":
    generate()
