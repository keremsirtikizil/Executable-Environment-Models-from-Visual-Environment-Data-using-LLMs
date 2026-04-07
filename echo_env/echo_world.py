"""
EchoWorld — a completely different invented grid environment.

Key differences from MagnetWorld:
  - Echo object moves in the OPPOSITE direction of the agent
  - Blocked echo does NOT cancel the agent's move
  - Void cells consume the echo (both vanish)
  - Beacon cells bounce the agent one extra step
"""

import copy
from PIL import Image, ImageDraw, ImageFont

EMPTY   = 0
WALL    = 1
ECHO    = 2   # green diamond — moves opposite to agent
VOID    = 3   # dark purple   — destroys echo on contact
BEACON  = 4   # yellow star   — bounces agent one extra step
AGENT   = 5

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

# (delta_row, delta_col) per action
_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# Opposite direction mapping
_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

CELL_SIZE    = 48
LABEL_HEIGHT = 24


class EchoWorld:
    def __init__(self, grid):
        self.reset(grid)

    def reset(self, grid):
        self.grid = copy.deepcopy(grid)
        self._sync_positions()

    def _sync_positions(self):
        self.agent_pos = None
        self.echo_pos = None
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == AGENT:
                    self.agent_pos = (r, c)
                elif val == ECHO:
                    self.echo_pos = (r, c)

    # ------------------------------------------------------------------
    def step(self, action):
        dr, dc = _DELTAS[action]
        ar, ac = self.agent_pos
        er, ec = self.echo_pos if self.echo_pos else (None, None)

        new_ar, new_ac = ar + dr, ac + dc

        # Rule 1: wall blocks agent
        if self.grid[new_ar][new_ac] == WALL:
            return

        # Check what agent is stepping onto (before overwrite)
        dest_cell = self.grid[new_ar][new_ac]

        # Move agent
        self.grid[ar][ac] = EMPTY
        self.grid[new_ar][new_ac] = AGENT
        self.agent_pos = (new_ar, new_ac)

        # Rule 2: Echo moves in OPPOSITE direction
        if er is not None:
            opp = _OPPOSITE[action]
            odr, odc = _DELTAS[opp]
            new_er, new_ec = er + odr, ec + odc

            # Rule 3: If echo would hit wall or agent, echo stays (agent already moved)
            if self.grid[new_er][new_ec] in (WALL, AGENT):
                pass  # echo stays
            # Rule 4: Void consumes echo — both vanish
            elif self.grid[new_er][new_ec] == VOID:
                self.grid[er][ec] = EMPTY
                self.grid[new_er][new_ec] = EMPTY  # void consumed
                self.echo_pos = None
            else:
                # Echo moves
                self.grid[er][ec] = EMPTY
                self.grid[new_er][new_ec] = ECHO
                self.echo_pos = (new_er, new_ec)

        # Rule 5: Beacon bounces agent one extra step in same direction
        if dest_cell == BEACON:
            bounce_r, bounce_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            next_cell = self.grid[bounce_r][bounce_c]
            if next_cell != WALL and next_cell != ECHO:
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = EMPTY
                self.grid[bounce_r][bounce_c] = AGENT
                self.agent_pos = (bounce_r, bounce_c)

    # ------------------------------------------------------------------
    def render_with_label(self, label: str) -> Image.Image:
        rows = len(self.grid)
        cols = len(self.grid[0])
        img_w = cols * CELL_SIZE
        img_h = rows * CELL_SIZE + LABEL_HEIGHT

        img  = Image.new("RGBA", (img_w, img_h), (240, 235, 245, 255))
        draw = ImageDraw.Draw(img)

        # Label bar
        draw.rectangle([0, 0, img_w, LABEL_HEIGHT], fill=(35, 30, 50, 255))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 5), label, fill=(255, 255, 255, 255), font=font)

        # Cells
        for r in range(rows):
            for c in range(cols):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE + LABEL_HEIGHT
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                _draw_cell(draw, self.grid[r][c], x0, y0, x1, y1)

        return img.convert("RGB")


# ------------------------------------------------------------------
def _draw_cell(draw, cell, x0, y0, x1, y1):
    pad = 6
    bg = (240, 235, 245)
    outline = (210, 205, 215)

    if cell == EMPTY:
        draw.rectangle([x0, y0, x1, y1], fill=bg)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=outline)

    elif cell == WALL:
        draw.rectangle([x0, y0, x1, y1], fill=(50, 45, 60))
        for y in range(y0 + 4, y1, 6):
            draw.line([(x0 + 2, y), (x1 - 2, y)], fill=(70, 65, 80), width=1)
        for x in range(x0 + 4, x1, 6):
            draw.line([(x, y0 + 2), (x, y1 - 2)], fill=(70, 65, 80), width=1)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(30, 25, 40))

    elif cell == ECHO:
        # Green diamond
        draw.rectangle([x0, y0, x1, y1], fill=bg)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=outline)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = CELL_SIZE // 2 - pad
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, fill=(50, 180, 80), outline=(30, 130, 50))

    elif cell == VOID:
        # Dark purple with X pattern
        draw.rectangle([x0, y0, x1, y1], fill=(55, 30, 70))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(80, 50, 100))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = CELL_SIZE // 2 - pad
        draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=(160, 80, 200), width=3)
        draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill=(160, 80, 200), width=3)

    elif cell == BEACON:
        # Yellow star (4-pointed)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 250, 220))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 200, 150))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = CELL_SIZE // 2 - pad
        ri = r // 3  # inner radius
        # 4-pointed star
        pts = [
            (cx, cy - r),     (cx + ri, cy - ri),
            (cx + r, cy),     (cx + ri, cy + ri),
            (cx, cy + r),     (cx - ri, cy + ri),
            (cx - r, cy),     (cx - ri, cy - ri),
        ]
        draw.polygon(pts, fill=(240, 200, 50), outline=(200, 160, 20))

    elif cell == AGENT:
        draw.rectangle([x0, y0, x1, y1], fill=bg)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=outline)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = CELL_SIZE // 2 - pad
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(210, 60, 60),
            outline=(160, 30, 30),
        )
