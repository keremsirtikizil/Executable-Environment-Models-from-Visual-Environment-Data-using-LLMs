"""
Generate a rule priority flowchart for PURE.
Outputs: results/rule_priority_diagram.png
"""

from PIL import Image, ImageDraw, ImageFont
import os


W, H = 1160, 720
BG = (255, 255, 255)

# Colors
C_CHECK   = (234, 241, 252)   # blue — decision
C_CHECK_S = (60, 90, 155)
C_ACTION  = (255, 241, 218)   # orange — rule action
C_ACTION_S= (190, 120, 30)
C_CANCEL  = (252, 232, 232)   # red — cancel
C_CANCEL_S= (175, 45, 45)
C_OK      = (224, 245, 224)   # green — success
C_OK_S    = (50, 130, 60)
C_ARROW   = (70, 70, 70)
C_TEXT    = (25, 25, 25)
C_SUB     = (90, 90, 90)
C_YES     = (50, 130, 60)
C_NO      = (175, 45, 45)
C_TITLE   = (25, 40, 95)
C_DIV     = (195, 195, 195)


def _font(size):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

FT = _font(20)
FH = _font(13)
FB = _font(12)
FS = _font(11)
FL = _font(10)


def diamond(d, cx, cy, w, h, fill, stroke, text, font=FB):
    pts = [(cx, cy - h//2), (cx + w//2, cy), (cx, cy + h//2), (cx - w//2, cy)]
    d.polygon(pts, fill=fill, outline=stroke, width=2)
    bb = d.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    d.text((cx - tw // 2, cy - th // 2), text, fill=stroke, font=font)


def rbox(d, cx, cy, w, h, fill, stroke, lines, font=FB):
    x0, y0 = cx - w // 2, cy - h // 2
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8,
                        fill=fill, outline=stroke, width=2)
    ty = y0 + 8
    for line, f, color in lines:
        bb = d.textbbox((0, 0), line, font=f)
        tw = bb[2] - bb[0]
        d.text((cx - tw // 2, ty), line, fill=color, font=f)
        ty += (bb[3] - bb[1]) + 4


def arrow_d(d, x, y0, y1):
    d.line([(x, y0), (x, y1 - 8)], fill=C_ARROW, width=2)
    d.polygon([(x, y1), (x - 5, y1 - 9), (x + 5, y1 - 9)], fill=C_ARROW)


def arrow_r(d, x0, y, x1):
    d.line([(x0, y), (x1 - 8, y)], fill=C_ARROW, width=2)
    d.polygon([(x1, y), (x1 - 9, y - 5), (x1 - 9, y + 5)], fill=C_ARROW)


def arrow_l(d, x0, y, x1):
    d.line([(x0, y), (x1 + 8, y)], fill=C_ARROW, width=2)
    d.polygon([(x1, y), (x1 + 9, y - 5), (x1 + 9, y + 5)], fill=C_ARROW)


def label(d, text, x, y, color, font=FL):
    d.text((x, y), text, fill=color, font=font)


def generate():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Title
    bb = d.textbbox((0, 0), "Rule Priority Flowchart", font=FT)
    d.text((W // 2 - (bb[2] - bb[0]) // 2, 16), "Rule Priority Flowchart",
           fill=C_TITLE, font=FT)
    bb2 = d.textbbox((0, 0), "MagnetWorld step(action) logic", font=FS)
    d.text((W // 2 - (bb2[2] - bb2[0]) // 2, 44), "MagnetWorld step(action) logic",
           fill=C_SUB, font=FS)
    d.line([(40, 68), (W - 40, 68)], fill=C_DIV, width=1)

    cx = W // 2
    # Positions
    y_wall = 110
    y_toward = 210
    y_metal_wall = 310
    y_metal_hole = 410
    y_both = 490
    y_agent = 310
    y_ice = 410

    right_x = cx + 280
    left_x = cx - 280

    # ── Diamond: Wall? ──────────────────────────────────────────────
    diamond(d, cx, y_wall, 240, 60, C_CHECK, C_CHECK_S, "Destination = WALL?")
    # YES → cancel
    arrow_r(d, cx + 120, y_wall, right_x - 80)
    label(d, "YES", cx + 125, y_wall - 16, C_YES)
    rbox(d, right_x, y_wall, 150, 44, C_CANCEL, C_CANCEL_S,
         [("Do nothing", FH, C_CANCEL_S), ("Rule 1", FS, C_SUB)])
    # NO ↓
    arrow_d(d, cx, y_wall + 30, y_toward - 30)
    label(d, "NO", cx + 6, y_wall + 34, C_NO)

    # ── Diamond: Toward? ────────────────────────────────────────────
    diamond(d, cx, y_toward, 260, 60, C_CHECK, C_CHECK_S, "Action toward metal?")

    # ── YES branch (left) ───────────────────────────────────────────
    arrow_l(d, cx - 130, y_toward, left_x + 80)
    label(d, "YES", cx - 165, y_toward - 16, C_YES)

    # Diamond: metal dest = wall?
    diamond(d, left_x, y_metal_wall, 260, 56, C_CHECK, C_CHECK_S, "Metal dest = WALL?")
    arrow_d(d, left_x, y_toward + 30, y_metal_wall - 28)

    # YES → cancel
    rbox(d, 90, y_metal_wall, 140, 44, C_CANCEL, C_CANCEL_S,
         [("Cancel both", FH, C_CANCEL_S), ("Rule 3", FS, C_SUB)])
    arrow_l(d, left_x - 130, y_metal_wall, 160)
    label(d, "YES", left_x - 126, y_metal_wall - 16, C_YES)

    # NO ↓
    arrow_d(d, left_x, y_metal_wall + 28, y_metal_hole - 28)
    label(d, "NO", left_x + 6, y_metal_wall + 30, C_NO)

    # Diamond: metal dest = hole?
    diamond(d, left_x, y_metal_hole, 260, 56, C_CHECK, C_CHECK_S, "Metal dest = HOLE?")

    # YES → consume
    rbox(d, 90, y_metal_hole, 155, 54, C_ACTION, C_ACTION_S,
         [("Agent moves", FH, C_ACTION_S), ("Metal + hole vanish", FS, C_ACTION_S), ("Rule 5", FS, C_SUB)])
    arrow_l(d, left_x - 130, y_metal_hole, 168)
    label(d, "YES", left_x - 126, y_metal_hole - 16, C_YES)

    # NO → both move
    arrow_d(d, left_x, y_metal_hole + 28, y_both - 22)
    label(d, "NO", left_x + 6, y_metal_hole + 30, C_NO)
    rbox(d, left_x, y_both, 180, 44, C_OK, C_OK_S,
         [("Both move 1 step", FH, C_OK_S), ("Rule 2", FS, C_SUB)])

    # ── NO branch (right): not toward ──────────────────────────────
    arrow_r(d, cx + 130, y_toward, right_x - 80)
    label(d, "NO", cx + 135, y_toward - 16, C_NO)

    rbox(d, right_x, y_toward + 70, 170, 44, C_OK, C_OK_S,
         [("Agent moves alone", FH, C_OK_S)])
    arrow_d(d, right_x, y_toward + 30, y_toward + 70 - 22)

    # Diamond: landed on ice?
    diamond(d, right_x, y_ice, 220, 56, C_CHECK, C_CHECK_S, "Landed on ICE?")
    arrow_d(d, right_x, y_toward + 70 + 22, y_ice - 28)

    # YES → slide
    slide_x = W - 100
    rbox(d, slide_x, y_ice, 150, 48, C_ACTION, C_ACTION_S,
         [("Agent slides", FH, C_ACTION_S), ("Rule 4", FS, C_SUB)])
    arrow_r(d, right_x + 110, y_ice, slide_x - 75)
    label(d, "YES", right_x + 115, y_ice - 16, C_YES)

    # NO → done
    rbox(d, right_x, y_ice + 80, 100, 34, (240, 240, 240), C_SUB,
         [("Done", FH, C_SUB)])
    arrow_d(d, right_x, y_ice + 28, y_ice + 80 - 17)
    label(d, "NO", right_x + 6, y_ice + 30, C_NO)

    # ── Legend ──────────────────────────────────────────────────────
    ly = H - 50
    d.line([(40, ly - 10), (W - 40, ly - 10)], fill=C_DIV, width=1)
    items = [
        (C_CHECK, C_CHECK_S, "Decision"),
        (C_OK, C_OK_S, "Move succeeds"),
        (C_ACTION, C_ACTION_S, "Special rule"),
        (C_CANCEL, C_CANCEL_S, "Move cancelled"),
    ]
    lx = 60
    for fill, stroke, lbl in items:
        d.rounded_rectangle([lx, ly, lx + 18, ly + 14], radius=4,
                            fill=fill, outline=stroke, width=1)
        d.text((lx + 24, ly), lbl, fill=C_TEXT, font=FS)
        lx += 200

    os.makedirs("results", exist_ok=True)
    out = "results/rule_priority_diagram.png"
    img.save(out, dpi=(300, 300))
    print(f"Rule priority diagram saved \u2192 {out}")
    return out


if __name__ == "__main__":
    generate()
