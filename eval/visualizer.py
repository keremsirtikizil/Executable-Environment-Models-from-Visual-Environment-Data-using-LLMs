"""
Visual report generator for PURE rule extraction results.

For each test case renders three side-by-side panels:
  INPUT (before)  |  EXPECTED (correct after)  |  GOT (LLM output)

GOT panel has a green border on PASS, red border on FAIL.
"""

import copy
from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ────────────────────────────────────────────────────────
TC  = 30          # thumbnail cell size (px)
TLH = 20          # thumbnail label bar height (px)
PAD = 14          # outer image padding
GAP = 10          # gap between panels / rows
LABEL_W = 200     # width of the row-label column

# Cell values
_EMPTY = 0; _WALL = 1; _METAL = 2; _ICE = 3; _HOLE = 4; _AGENT = 5

# ── Colors ──────────────────────────────────────────────────────────────────
BG_PAGE       = (232, 230, 226)
BG_HEADER     = (30,  30,  30)
BG_ROW_PASS   = (225, 242, 225)
BG_ROW_FAIL   = (242, 225, 225)
BG_CELL_EMPTY = (245, 240, 225)
BG_WALL       = (55,  55,  55)
BG_WALL_HATCH = (78,  78,  78)
C_METAL_FILL  = (70,  130, 200)
C_METAL_OUT   = (40,  90,  160)
C_AGENT_FILL  = (210, 60,  60)
C_AGENT_OUT   = (160, 30,  30)
C_CELL_OUT    = (200, 195, 185)
C_WALL_OUT    = (35,  35,  35)
C_PASS        = (30,  155, 30)
C_FAIL        = (200, 40,  40)
C_LABEL_PASS  = (20,  120, 20)
C_LABEL_FAIL  = (180, 30,  30)


# ── Font loader ─────────────────────────────────────────────────────────────
def _font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Cell renderer ───────────────────────────────────────────────────────────
def _draw_cell(draw, val, x0, y0, x1, y1):
    pad = max(3, TC // 7)
    if val == _WALL:
        draw.rectangle([x0, y0, x1, y1], fill=BG_WALL)
        step = max(4, TC // 6)
        for y in range(y0 + 2, y1, step):
            draw.line([(x0 + 1, y), (x1 - 1, y)], fill=BG_WALL_HATCH, width=1)
        for x in range(x0 + 2, x1, step):
            draw.line([(x, y0 + 1), (x, y1 - 1)], fill=BG_WALL_HATCH, width=1)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=C_WALL_OUT)
    elif val == _EMPTY:
        draw.rectangle([x0, y0, x1, y1], fill=BG_CELL_EMPTY)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=C_CELL_OUT)
    elif val == _METAL:
        draw.rectangle([x0, y0, x1, y1], fill=BG_CELL_EMPTY)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=C_CELL_OUT)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = TC // 2 - pad
        if r > 1:
            pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw.polygon(pts, fill=C_METAL_FILL, outline=C_METAL_OUT)
    elif val == _ICE:
        draw.rectangle([x0, y0, x1, y1], fill=(200, 230, 245))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(150, 200, 225))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = TC // 2 - pad
        if r > 1:
            pts = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
            draw.polygon(pts, fill=(60, 190, 130), outline=(30, 140, 90))
    elif val == _HOLE:
        draw.rectangle([x0, y0, x1, y1], fill=(60, 50, 45))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(90, 75, 65))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = TC // 2 - pad
        if r > 1:
            draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=(230, 140, 40), width=2)
            draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill=(230, 140, 40), width=2)
    elif val == _AGENT:
        draw.rectangle([x0, y0, x1, y1], fill=BG_CELL_EMPTY)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=C_CELL_OUT)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = TC // 2 - pad
        if r > 1:
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=C_AGENT_FILL, outline=C_AGENT_OUT,
            )


# ── Grid renderer ────────────────────────────────────────────────────────────
def _render_grid(grid, label=""):
    rows, cols = len(grid), len(grid[0])
    w, h = cols * TC, rows * TC + TLH
    img = Image.new("RGB", (w, h), BG_CELL_EMPTY)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, TLH], fill=(50, 50, 50))
    draw.text((4, 3), label, fill=(230, 230, 230), font=_font(11))
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * TC, r * TC + TLH
            _draw_cell(draw, grid[r][c], x0, y0, x0 + TC, y0 + TC)
    return img


# ── Utilities ────────────────────────────────────────────────────────────────
def _pad_to(img, w, h, bg=BG_PAGE):
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def _border(img, color, width=3):
    draw = ImageDraw.Draw(img)
    for i in range(width):
        draw.rectangle([i, i, img.width - 1 - i, img.height - 1 - i], outline=color)
    return img


def _reconstruct(base_grid, agent_pos, metal_pos, consumed_holes=None):
    grid = copy.deepcopy(base_grid)
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] in (_AGENT, _METAL):
                grid[r][c] = _EMPTY
    # Remove consumed holes (metal fell in)
    if consumed_holes:
        for hr, hc in consumed_holes:
            grid[hr][hc] = _EMPTY
    if agent_pos:
        grid[agent_pos[0]][agent_pos[1]] = _AGENT
    if metal_pos:
        grid[metal_pos[0]][metal_pos[1]] = _METAL
    return grid


# ── Main entry point ─────────────────────────────────────────────────────────
def render_results(eval_result: dict, test_cases: list, output_path: str, title: str = ""):
    """
    Render a full visual report as a PNG image.

    eval_result  — returned by evaluate_extracted_function()
    test_cases   — returned by build_test_cases()
    output_path  — where to save the PNG
    title        — e.g. "fal/google/gemini-2.5-pro"
    """
    by_name = {r["name"]: r for r in eval_result.get("results", [])}

    # Panel dimensions based on largest grid in the suite
    max_rows = max(len(tc["grid"])    for tc in test_cases)
    max_cols = max(len(tc["grid"][0]) for tc in test_cases)
    panel_w  = max_cols * TC
    panel_h  = max_rows * TC + TLH

    row_h    = panel_h + GAP
    n        = len(test_cases)

    header_h   = 64
    col_head_h = 22
    total_w = PAD + LABEL_W + GAP + (panel_w + GAP) * 3 + PAD
    total_h = PAD + header_h + col_head_h + n * row_h + PAD

    img  = Image.new("RGB", (total_w, total_h), BG_PAGE)
    draw = ImageDraw.Draw(img)

    # ── Header bar ──────────────────────────────────────────────────────────
    draw.rectangle([0, 0, total_w, PAD + header_h], fill=BG_HEADER)
    draw.text((PAD, PAD + 6),  "PURE — Rule Extraction Visual Report",
              fill=(255, 255, 255), font=_font(17))
    n_p = eval_result.get("n_passed", 0)
    n_t = eval_result.get("n_total",  0)
    acc = eval_result.get("accuracy", 0.0) * 100
    draw.text((PAD, PAD + 30),
              f"{title}     Passed: {n_p}/{n_t}  ({acc:.1f}%)",
              fill=(185, 185, 185), font=_font(13))

    # ── Column headers ───────────────────────────────────────────────────────
    y_ch    = PAD + header_h + 4
    x_in    = PAD + LABEL_W + GAP
    x_exp   = x_in  + panel_w + GAP
    x_got   = x_exp + panel_w + GAP
    for x, lbl in [(x_in, "INPUT (before)"),
                   (x_exp, "EXPECTED"),
                   (x_got, "GOT (LLM output)")]:
        draw.text((x + 4, y_ch), lbl, fill=(90, 90, 90), font=_font(11))

    # ── Test-case rows ────────────────────────────────────────────────────────
    y_base = PAD + header_h + col_head_h
    fnt_name = _font(11)
    fnt_desc = _font(10)

    for i, tc in enumerate(test_cases):
        y = y_base + i * row_h
        r = by_name.get(tc["name"])

        passed = r["passed"] if (r and "got_agent" in r) else False
        status = "PASS" if passed else "FAIL"
        row_bg = BG_ROW_PASS if passed else BG_ROW_FAIL
        lbl_c  = C_LABEL_PASS if passed else C_LABEL_FAIL
        bdr_c  = C_PASS       if passed else C_FAIL

        # Row background
        draw.rectangle([PAD, y, total_w - PAD, y + panel_h + 2], fill=row_bg)

        # Row label
        draw.text((PAD + 4, y + 4),  f"[{status}]",       fill=lbl_c,    font=fnt_name)
        draw.text((PAD + 4, y + 18), tc["name"],           fill=(50,50,50), font=fnt_name)
        desc = tc.get("description", "")
        if len(desc) > 30:
            desc = desc[:27] + "…"
        draw.text((PAD + 4, y + 32), desc, fill=(110,110,110), font=fnt_desc)

        # INPUT panel
        in_img = _render_grid(tc["grid"], label="before")
        in_img = _pad_to(in_img, panel_w, panel_h)
        img.paste(in_img, (x_in, y))

        # EXPECTED panel
        exp_grid = _reconstruct(
            tc["grid"], tc["expected_agent_pos"], tc["expected_metal_pos"],
            consumed_holes=tc.get("consumed_holes")
        )
        exp_img = _render_grid(exp_grid, label="expected")
        exp_img = _pad_to(exp_img, panel_w, panel_h)
        _border(exp_img, (80, 160, 80), width=2)
        img.paste(exp_img, (x_exp, y))

        # GOT panel
        if r and "got_agent" in r:
            got_grid = _reconstruct(tc["grid"], r["got_agent"], r["got_metal"],
                                       consumed_holes=tc.get("consumed_holes"))
            got_img  = _render_grid(got_grid, label="got")
        elif r and r.get("error"):
            got_img = Image.new("RGB", (panel_w, panel_h), (255, 220, 220))
            d2 = ImageDraw.Draw(got_img)
            d2.text((4, 4), "ERROR", fill=(180, 0, 0), font=fnt_name)
            err_msg = str(r["error"])[:40]
            d2.text((4, 20), err_msg, fill=(150, 0, 0), font=fnt_desc)
        else:
            got_img = Image.new("RGB", (panel_w, panel_h), (220, 220, 220))
            d2 = ImageDraw.Draw(got_img)
            d2.text((4, 4), "no result", fill=(120, 120, 120), font=fnt_name)

        got_img = _pad_to(got_img, panel_w, panel_h)
        _border(got_img, bdr_c, width=3)
        img.paste(got_img, (x_got, y))

    img.save(output_path)
    print(f"  Visualization → {output_path}")
