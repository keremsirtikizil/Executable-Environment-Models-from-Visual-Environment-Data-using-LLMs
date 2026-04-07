import copy
from PIL import Image, ImageDraw, ImageFont

EMPTY = 0
WALL  = 1
METAL = 2
ICE   = 3
HOLE  = 4
AGENT = 5

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

# (delta_row, delta_col) per action
_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

CELL_SIZE    = 48
LABEL_HEIGHT = 24


class MagnetWorld:
    def __init__(self, grid):
        self.reset(grid)

    # ------------------------------------------------------------------
    def reset(self, grid):
        self.grid = copy.deepcopy(grid)
        self._sync_positions()

    def _sync_positions(self):
        self.agent_pos = None
        self.metal_pos = None
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == AGENT:
                    self.agent_pos = (r, c)
                elif val == METAL:
                    self.metal_pos = (r, c)

    # ------------------------------------------------------------------
    def step(self, action):
        dr, dc = _DELTAS[action]
        ar, ac = self.agent_pos
        mr, mc = self.metal_pos if self.metal_pos else (None, None)

        new_ar, new_ac = ar + dr, ac + dc

        # Rule 1: wall blocks agent
        if self.grid[new_ar][new_ac] == WALL:
            return

        # Is agent moving toward metal along the action axis?
        toward = False
        if mr is not None:
            toward = (
                (dr ==  1 and mr > ar) or
                (dr == -1 and mr < ar) or
                (dc ==  1 and mc > ac) or
                (dc == -1 and mc < ac)
            )

        # Check what the agent is stepping onto (before overwrite)
        dest_cell = self.grid[new_ar][new_ac]

        if toward:
            new_mr, new_mc = mr + dr, mc + dc
            # Rule 3: metal would hit wall — cancel entire move
            if self.grid[new_mr][new_mc] == WALL:
                return
            # Rule 5: metal pushed into hole — metal and hole both vanish
            if self.grid[new_mr][new_mc] == HOLE:
                self.grid[ar][ac]         = EMPTY
                self.grid[mr][mc]         = EMPTY
                self.grid[new_ar][new_ac] = AGENT
                self.grid[new_mr][new_mc] = EMPTY   # hole consumed
                self.agent_pos = (new_ar, new_ac)
                self.metal_pos = None
            else:
                # Rule 2: both move
                self.grid[ar][ac]         = EMPTY
                self.grid[mr][mc]         = EMPTY
                self.grid[new_ar][new_ac] = AGENT
                self.grid[new_mr][new_mc] = METAL
                self.agent_pos = (new_ar, new_ac)
                self.metal_pos = (new_mr, new_mc)
        else:
            # Only agent moves
            self.grid[ar][ac]         = EMPTY
            self.grid[new_ar][new_ac] = AGENT
            self.agent_pos = (new_ar, new_ac)

        # Rule 4: ICE — agent slides until hitting non-ice cell
        # Only applies when agent is NOT doing an attraction move
        if not toward and dest_cell == ICE:
            while True:
                slide_r = self.agent_pos[0] + dr
                slide_c = self.agent_pos[1] + dc
                next_cell = self.grid[slide_r][slide_c]
                if next_cell == WALL or next_cell == METAL:
                    break
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = EMPTY
                self.grid[slide_r][slide_c] = AGENT
                self.agent_pos = (slide_r, slide_c)
                if next_cell != ICE:
                    break

    # ------------------------------------------------------------------
    def render_with_label(self, label: str) -> Image.Image:
        rows = len(self.grid)
        cols = len(self.grid[0])
        img_w = cols * CELL_SIZE
        img_h = rows * CELL_SIZE + LABEL_HEIGHT

        img  = Image.new("RGBA", (img_w, img_h), (245, 240, 225, 255))
        draw = ImageDraw.Draw(img)

        # Label bar
        draw.rectangle([0, 0, img_w, LABEL_HEIGHT], fill=(45, 45, 45, 255))
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
    if cell == EMPTY:
        draw.rectangle([x0, y0, x1, y1], fill=(245, 240, 225))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(210, 205, 190))

    elif cell == WALL:
        draw.rectangle([x0, y0, x1, y1], fill=(55, 55, 55))
        # Crosshatch: horizontal + vertical lines, clipped to cell interior
        for y in range(y0 + 4, y1, 6):
            draw.line([(x0 + 2, y), (x1 - 2, y)], fill=(75, 75, 75), width=1)
        for x in range(x0 + 4, x1, 6):
            draw.line([(x, y0 + 2), (x, y1 - 2)], fill=(75, 75, 75), width=1)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(35, 35, 35))

    elif cell == METAL:
        draw.rectangle([x0, y0, x1, y1], fill=(245, 240, 225))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(210, 205, 190))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r  = CELL_SIZE // 2 - pad
        # Blue diamond (rotated square)
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, fill=(70, 130, 200), outline=(40, 90, 160))

    elif cell == ICE:
        # Light cyan background with snowflake-like pattern
        draw.rectangle([x0, y0, x1, y1], fill=(200, 230, 245))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(150, 200, 225))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r  = CELL_SIZE // 2 - pad
        # Green triangle pointing up
        pts = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        draw.polygon(pts, fill=(60, 190, 130), outline=(30, 140, 90))

    elif cell == HOLE:
        # Dark background with orange X
        draw.rectangle([x0, y0, x1, y1], fill=(60, 50, 45))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(90, 75, 65))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r  = CELL_SIZE // 2 - pad
        draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=(230, 140, 40), width=3)
        draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill=(230, 140, 40), width=3)

    elif cell == AGENT:
        draw.rectangle([x0, y0, x1, y1], fill=(245, 240, 225))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(210, 205, 190))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r  = CELL_SIZE // 2 - pad
        # Red circle
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(210, 60, 60),
            outline=(160, 30, 30),
        )
