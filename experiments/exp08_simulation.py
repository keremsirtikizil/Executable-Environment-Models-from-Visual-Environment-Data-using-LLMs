"""
Experiment 08 — Pseudocode → Direct Simulation + Visualization.

Implements three simulations written directly from PURE pseudocode output,
without any LLM at render time. Each simulation faithfully follows the state
variables and IF/THEN rules extracted by PURE.

Scenes:
  bouncing_ball   — exp05 (CORRECT) — droplet on superhydrophobic surface
  bowling_pins    — exp06 (PARTIALLY CORRECT) — full bowling game with pinsetter
  billiard_break  — exp06 — break shot (elastic 2D ball-ball collisions)

Output per scene:
  results/exp08/<scene>/frame_NNN.png   — individual frames
  results/exp08/<scene>/simulation.gif  — animated GIF

Run from project root:
  python experiments/exp08_simulation.py
"""

import os
import sys
import math
import glob
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Circle, FancyArrow

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

RESULTS_ROOT = "results/exp08"
N_STEPS      = 180    # frames per simulation
GIF_DURATION = 50     # ms per frame in GIF
FIG_DPI      = 100
FIG_SIZE     = (8, 6)


# ═══════════════════════════════════════════════════════════════════
# 1. BOUNCING BALL  (from exp05 pseudocode — CORRECT verdict)
#
# STATE VARIABLES (from pseudocode):
#   droplet.y, droplet.velocity_y
#   droplet.height, droplet.width, droplet.initial_height, droplet.min_height
#   droplet.state: [AIRBORNE, COMPRESSING, REBOUNDING]
#   surface.y, gravity, deformation_rate, restitution_velocity
# ═══════════════════════════════════════════════════════════════════

class BouncingBallSim:
    def __init__(self):
        self.surface_y       = 0.08
        self.initial_height  = 0.07
        self.min_height      = 0.025

        # droplet position (centre)
        self.x     = 0.50
        self.y     = 0.82
        self.vy    = 0.0

        self.height = self.initial_height
        self.width  = self.initial_height

        self.state         = "AIRBORNE"
        self.gravity       = 0.012
        self.deform_rate   = 0.006
        self._bounce       = 0          # counts completed bounces (for energy loss)
        self._base_restitution = 0.55   # fraction of impact speed recovered

    def step(self):
        if self.state == "AIRBORNE":
            self.vy -= self.gravity
            self.y  += self.vy
            if self.y <= self.surface_y and self.vy < 0:
                self._impact_speed = abs(self.vy)
                self.y   = self.surface_y
                self.vy  = 0.0
                self.state = "COMPRESSING"

        elif self.state == "COMPRESSING":
            self.height -= self.deform_rate
            self.width  += self.deform_rate * 1.4   # conserve volume (approx)
            if self.height <= self.min_height:
                self.state = "REBOUNDING"

        elif self.state == "REBOUNDING":
            self.height += self.deform_rate
            self.width  -= self.deform_rate * 1.4
            if self.height >= self.initial_height:
                self.height = self.initial_height
                self.width  = self.initial_height
                self.state  = "AIRBORNE"
                # energy loss: restitution coefficient applied to each bounce
                self.vy = self._impact_speed * (self._base_restitution ** (self._bounce + 1))
                self._bounce += 1

        # Reset once bounces die out or ball escapes top
        if self._bounce > 7 or self.y > 1.05:
            self.__init__()

    def get_state(self):
        return dict(x=self.x, y=self.y, height=self.height, width=self.width,
                    state=self.state, bounce=self._bounce)


def render_bouncing_ball(sim_state, ax, step):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.axis("off")

    # surface
    ax.axhline(y=0.08, color="#4ecdc4", linewidth=2.5, zorder=1)
    ax.text(0.5, 0.04, "superhydrophobic surface", ha="center", va="center",
            color="#4ecdc4", fontsize=8, zorder=2)

    s  = sim_state["state"]
    cx = sim_state["x"]
    cy = sim_state["y"]
    h  = sim_state["height"]
    w  = sim_state["width"]

    # colour by state
    colour = {"AIRBORNE": "#74b9ff", "COMPRESSING": "#fd79a8", "REBOUNDING": "#55efc4"}[s]

    el = Ellipse((cx, cy), width=w, height=h, color=colour, zorder=3)
    ax.add_patch(el)

    # faint motion trail
    ax.scatter([cx], [cy + h * 0.6], s=15, color=colour, alpha=0.3, zorder=2)

    ax.set_title(f"Bouncing Ball  |  step {step:03d}  |  state: {s}  |  bounce #{sim_state['bounce']}",
                 color="white", fontsize=9, pad=6)
    ax.set_facecolor("#1a1a2e")
    ax.figure.patch.set_facecolor("#1a1a2e")


# ═══════════════════════════════════════════════════════════════════
# 2. BOWLING PINS  (from exp06 pseudocode — PARTIALLY CORRECT)
#
# STATE VARIABLES (from pseudocode):
#   game_state: (ready, ball_rolling, pins_interacting, end_of_roll)
#   reset_cycle: (idle, preparing_spare, resetting_frame)
#   roll_number, game_is_over, initial_hit_type, ui_text
#   ball: (position, velocity)
#   pins[10]: (position, state: standing/falling/fallen/lifted, velocity)
#   sweep_arm, pinsetter_deck
# ═══════════════════════════════════════════════════════════════════

PIN_SPACING  = 0.075
BALL_RADIUS  = 0.038
PIN_RADIUS   = 0.022
BALL_SPEED   = 0.018
COLLISION_D  = BALL_RADIUS + PIN_RADIUS
PIN_COL_D    = PIN_RADIUS * 2.2


def _pin_positions():
    d = PIN_SPACING
    return [
        # row 1 (head)
        (0.500, 0.740),
        # row 2
        (0.500 - d * 0.5, 0.740 + d * 0.87),
        (0.500 + d * 0.5, 0.740 + d * 0.87),
        # row 3
        (0.500 - d,       0.740 + d * 1.74),
        (0.500,           0.740 + d * 1.74),
        (0.500 + d,       0.740 + d * 1.74),
        # row 4
        (0.500 - d * 1.5, 0.740 + d * 2.61),
        (0.500 - d * 0.5, 0.740 + d * 2.61),
        (0.500 + d * 0.5, 0.740 + d * 2.61),
        (0.500 + d * 1.5, 0.740 + d * 2.61),
    ]


class BowlingSim:
    def __init__(self):
        self._init_pins()
        self.ball_x  = 0.500
        self.ball_y  = 0.080
        self.ball_vx = 0.0
        self.ball_vy = BALL_SPEED
        self.ball_visible = True

        self.game_state   = "ready"
        self.reset_cycle  = "idle"
        self.roll_number  = 1
        self.game_is_over = False
        self.initial_hit_type = "none"
        self.ui_text      = ""

        # sweep arm (y position sweeping from 0.68 → 1.0)
        self.sweep_y      = 0.68
        self.sweep_active = False
        # pinsetter deck timer
        self._reset_timer = 0
        self._roll_done   = False
        self._frame       = 0

    def _init_pins(self):
        self.pins = []
        for (px, py) in _pin_positions():
            self.pins.append({
                "x": px, "y": py,
                "vx": 0.0, "vy": 0.0,
                "state": "standing",
                "init_x": px, "init_y": py,
            })

    def _dist(self, ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    def step(self):
        self._frame += 1

        # ── READY: launch ball ────────────────────────────────────
        if self.game_state == "ready":
            if self._frame % 60 == 1:     # auto-roll every 60 frames
                self.ball_x  = 0.500 + random.uniform(-0.03, 0.03)
                self.ball_y  = 0.080
                self.ball_vy = BALL_SPEED
                self.ball_vx = (0.500 - self.ball_x) * 0.05
                self.ball_visible = True
                self.game_state   = "ball_rolling"
                self.initial_hit_type = "none"
                self.ui_text = ""

        # ── BALL ROLLING ──────────────────────────────────────────
        elif self.game_state == "ball_rolling":
            self.ball_x += self.ball_vx
            self.ball_y += self.ball_vy

            for p in self.pins:
                if p["state"] == "standing":
                    if self._dist(self.ball_x, self.ball_y, p["x"], p["y"]) < COLLISION_D:
                        # first collision → classify hit type
                        if self.initial_hit_type == "none":
                            dx = self.ball_x - p["x"]
                            if dx < -0.01:
                                self.initial_hit_type = "brooklyn"
                            elif dx > 0.01:
                                self.initial_hit_type = "pocket"
                            else:
                                self.initial_hit_type = "nose_hit"

                        p["state"] = "falling"
                        # impart velocity: away from ball centre
                        angle = math.atan2(p["y"] - self.ball_y, p["x"] - self.ball_x)
                        speed = 0.025 + random.uniform(0, 0.01)
                        p["vx"] = math.cos(angle) * speed
                        p["vy"] = math.sin(angle) * speed
                        self.ball_vx *= 0.85
                        self.game_state = "pins_interacting"

            if self.ball_y > 1.05:
                self.ball_visible = False
                self.game_state   = "pins_interacting"

        # ── PINS INTERACTING ──────────────────────────────────────
        elif self.game_state == "pins_interacting":
            # move ball
            if self.ball_visible:
                self.ball_x += self.ball_vx
                self.ball_y += self.ball_vy
                if self.ball_y > 1.05:
                    self.ball_visible = False

            # move falling pins + cascade
            any_moving = False
            for p in self.pins:
                if p["state"] == "falling":
                    any_moving = True
                    p["x"] += p["vx"]
                    p["y"] += p["vy"]
                    p["vx"] *= 0.92
                    p["vy"] *= 0.92
                    if abs(p["vx"]) < 0.0008 and abs(p["vy"]) < 0.0008:
                        p["state"] = "fallen"
                    # cascade: knock into standing neighbours
                    for q in self.pins:
                        if q["state"] == "standing" and q is not p:
                            if self._dist(p["x"], p["y"], q["x"], q["y"]) < PIN_COL_D:
                                q["state"] = "falling"
                                angle = math.atan2(q["y"] - p["y"], q["x"] - p["x"])
                                speed = math.hypot(p["vx"], p["vy"]) * 0.7
                                q["vx"] = math.cos(angle) * speed
                                q["vy"] = math.sin(angle) * speed

            if not any_moving and not self.ball_visible:
                self.game_state = "end_of_roll"
                self._reset_timer = 0

        # ── END OF ROLL ───────────────────────────────────────────
        elif self.game_state == "end_of_roll":
            self._reset_timer += 1
            if self._reset_timer == 1:
                standing = [p for p in self.pins if p["state"] == "standing"]
                fallen   = [p for p in self.pins if p["state"] in ("fallen", "falling")]
                if self.roll_number == 1:
                    if len(standing) == 0:
                        self.ui_text = "Brooklyn Strike" if self.initial_hit_type == "brooklyn" else "Strike"
                        self.reset_cycle = "resetting_frame"
                    else:
                        self.reset_cycle = "preparing_spare"
                else:
                    self.ui_text = f"Spare! (+{len(fallen)})" if len(standing) == 0 else f"+{len(fallen)}"
                    self.reset_cycle = "resetting_frame"

            if self._reset_timer > 40:
                self._do_reset()

        return self.get_state()

    def _do_reset(self):
        # sweep all fallen pins, respawn standing ones, reset for next roll
        if self.reset_cycle == "preparing_spare":
            for p in self.pins:
                if p["state"] in ("fallen", "falling"):
                    p["state"] = "cleared"
            self.roll_number  = 2
        else:
            self._init_pins()
            self.roll_number  = 1
            self.game_is_over = (random.random() < 0.15)   # simulate 10th frame occasionally

        self.reset_cycle = "idle"
        self.game_state  = "ready"
        self._frame      = 0
        if self.game_is_over:
            self.game_is_over = False

    def get_state(self):
        return {
            "ball_x": self.ball_x, "ball_y": self.ball_y, "ball_visible": self.ball_visible,
            "pins": [(p["x"], p["y"], p["state"]) for p in self.pins],
            "game_state": self.game_state,
            "ui_text": self.ui_text,
            "hit_type": self.initial_hit_type,
        }


def render_bowling(state, ax, step):
    ax.set_xlim(0.2, 0.8); ax.set_ylim(0.0, 1.05)
    ax.set_facecolor("#2d1b00")
    ax.figure.patch.set_facecolor("#2d1b00")
    ax.axis("off")

    # lane
    lane = patches.Rectangle((0.30, 0.00), 0.40, 1.05,
                               linewidth=1.5, edgecolor="#c8a96e",
                               facecolor="#c8a96e22", zorder=1)
    ax.add_patch(lane)

    # foul line
    ax.axhline(y=0.12, color="#ff6b6b", linewidth=1.5, xmin=0.25, xmax=0.75, zorder=2)

    # ball
    if state["ball_visible"]:
        ball = Circle((state["ball_x"], state["ball_y"]), BALL_RADIUS,
                       color="#2c3e50", zorder=4)
        ax.add_patch(ball)
        ball_ring = Circle((state["ball_x"], state["ball_y"]), BALL_RADIUS,
                            fill=False, edgecolor="#95a5a6", linewidth=1.5, zorder=5)
        ax.add_patch(ball_ring)

    # pins
    pin_colours = {
        "standing": "#ecf0f1",
        "falling":  "#e74c3c",
        "fallen":   "#7f8c8d",
        "lifted":   "#f39c12",
        "cleared":  None,
    }
    for (px, py, pstate) in state["pins"]:
        col = pin_colours.get(pstate)
        if col is None:
            continue
        pin = Circle((px, py), PIN_RADIUS, color=col, zorder=3)
        ax.add_patch(pin)

    # ui text
    if state["ui_text"]:
        ax.text(0.50, 0.55, state["ui_text"], ha="center", va="center",
                color="#f1c40f", fontsize=18, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000088"), zorder=6)

    ax.set_title(
        f"Bowling  |  step {step:03d}  |  {state['game_state']}  |  hit: {state['hit_type']}",
        color="white", fontsize=9, pad=6)


# ═══════════════════════════════════════════════════════════════════
# 3. BILLIARD BREAK  (from exp06 — thermal camera video)
#
# Implements 2D elastic ball-ball collisions: a cue ball breaks a
# triangular rack of 15 balls. Physics faithful to the standard
# billiard break pseudocode extracted by PURE (velocities, collisions,
# cushion rebounds).
# ═══════════════════════════════════════════════════════════════════

TABLE_W = 1.0
TABLE_H = 0.55
BBALL_R = 0.022
FRICTION = 0.988   # per step speed multiplier


def _rack_positions():
    """15-ball triangle rack centred at (0.68, TABLE_H/2)."""
    cx, cy = 0.68, TABLE_H / 2
    d = BBALL_R * 2.1
    positions = []
    for row in range(5):
        for col in range(row + 1):
            x = cx + row * d * math.cos(math.pi / 6)
            y = cy + (col - row / 2) * d
            positions.append((x, y))
    return positions


BALL_COLOURS = [
    "#f1c40f",  # 1 yellow
    "#2980b9",  # 2 blue
    "#e74c3c",  # 3 red
    "#8e44ad",  # 4 purple
    "#e67e22",  # 5 orange
    "#27ae60",  # 6 green
    "#922b21",  # 7 maroon
    "#1a1a1a",  # 8 black
    "#f1c40f",  # 9 striped yellow
    "#2980b9",  # 10 striped blue
    "#e74c3c",  # 11 striped red
    "#8e44ad",  # 12 striped purple
    "#e67e22",  # 13 striped orange
    "#27ae60",  # 14 striped green
    "#922b21",  # 15 striped maroon
]


class BilliardBreakSim:
    def __init__(self):
        self._reset()

    def _reset(self):
        rack = _rack_positions()
        self.balls = []
        for i, (x, y) in enumerate(rack):
            self.balls.append({
                "x": x, "y": y, "vx": 0.0, "vy": 0.0,
                "colour": BALL_COLOURS[i], "active": True,
            })
        # cue ball
        self.balls.append({
            "x": 0.22, "y": TABLE_H / 2,
            "vx": 0.048, "vy": random.uniform(-0.003, 0.003),
            "colour": "#ffffff", "active": True,
        })
        self._step_count = 0

    def step(self):
        self._step_count += 1

        # move all balls
        for b in self.balls:
            if not b["active"]: continue
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["vx"] *= FRICTION
            b["vy"] *= FRICTION

            # cushion bounces
            if b["x"] - BBALL_R < 0:
                b["x"] = BBALL_R; b["vx"] = abs(b["vx"])
            if b["x"] + BBALL_R > TABLE_W:
                b["x"] = TABLE_W - BBALL_R; b["vx"] = -abs(b["vx"])
            if b["y"] - BBALL_R < 0:
                b["y"] = BBALL_R; b["vy"] = abs(b["vy"])
            if b["y"] + BBALL_R > TABLE_H:
                b["y"] = TABLE_H - BBALL_R; b["vy"] = -abs(b["vy"])

        # elastic ball-ball collisions
        n = len(self.balls)
        for i in range(n):
            if not self.balls[i]["active"]: continue
            for j in range(i + 1, n):
                if not self.balls[j]["active"]: continue
                a, b = self.balls[i], self.balls[j]
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                dist = math.hypot(dx, dy)
                if dist < BBALL_R * 2 and dist > 1e-9:
                    # unit normal
                    nx, ny = dx / dist, dy / dist
                    # overlap correction
                    overlap = BBALL_R * 2 - dist
                    a["x"] -= nx * overlap / 2
                    a["y"] -= ny * overlap / 2
                    b["x"] += nx * overlap / 2
                    b["y"] += ny * overlap / 2
                    # relative velocity along normal
                    dvn = (b["vx"] - a["vx"]) * nx + (b["vy"] - a["vy"]) * ny
                    if dvn < 0:  # approaching
                        a["vx"] += dvn * nx
                        a["vy"] += dvn * ny
                        b["vx"] -= dvn * nx
                        b["vy"] -= dvn * ny

        # reset once everything stops or 300 steps passed
        speeds = [math.hypot(b["vx"], b["vy"]) for b in self.balls if b["active"]]
        if self._step_count > 300 or (self._step_count > 60 and max(speeds, default=0) < 0.0005):
            self._reset()

    def get_state(self):
        return {"balls": [(b["x"], b["y"], b["colour"], b["active"]) for b in self.balls],
                "step": self._step_count}


def render_billiards(state, ax, step):
    ax.set_xlim(-0.02, TABLE_W + 0.02)
    ax.set_ylim(-0.02, TABLE_H + 0.02)
    ax.set_facecolor("#1a0a00")
    ax.figure.patch.set_facecolor("#1a0a00")
    ax.set_aspect("equal")
    ax.axis("off")

    # table felt
    table = patches.Rectangle((0, 0), TABLE_W, TABLE_H,
                                linewidth=3, edgecolor="#8B4513",
                                facecolor="#1a6b3a", zorder=1)
    ax.add_patch(table)

    # corner pockets (simple circles)
    for (px, py) in [(0, 0), (0, TABLE_H), (TABLE_W, 0), (TABLE_W, TABLE_H),
                     (TABLE_W / 2, 0), (TABLE_W / 2, TABLE_H)]:
        pocket = Circle((px, py), 0.025, color="#111111", zorder=2)
        ax.add_patch(pocket)

    for (bx, by, col, active) in state["balls"]:
        if not active: continue
        ball = Circle((bx, by), BBALL_R, color=col, zorder=3)
        ax.add_patch(ball)
        ring = Circle((bx, by), BBALL_R, fill=False,
                       edgecolor="#ffffff44", linewidth=0.8, zorder=4)
        ax.add_patch(ring)

    ax.set_title(f"Billiard Break  |  step {step:03d}", color="white", fontsize=9, pad=6)


# ═══════════════════════════════════════════════════════════════════
# Runner — renders frames and builds GIF
# ═══════════════════════════════════════════════════════════════════

SCENES = [
    ("bouncing_ball",  BouncingBallSim,    render_bouncing_ball),
    ("bowling_pins",   BowlingSim,         render_bowling),
    ("billiard_break", BilliardBreakSim,   render_billiards),
]


def save_frames_and_gif(scene_name, sim_cls, render_fn):
    out_dir = os.path.join(RESULTS_ROOT, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    if existing:
        print(f"  [SKIP] {len(existing)} frames already exist in {out_dir}")
        return

    sim = sim_cls()
    frame_paths = []

    print(f"  Rendering {N_STEPS} frames...")
    for i in range(N_STEPS):
        sim.step()
        state = sim.get_state()

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        render_fn(state, ax, i)
        plt.tight_layout(pad=0.5)

        path = os.path.join(out_dir, f"frame_{i:03d}.png")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(path)

        if (i + 1) % 30 == 0:
            print(f"    {i + 1}/{N_STEPS}")

    gif_path = os.path.join(out_dir, "simulation.gif")
    imgs = [Image.open(p) for p in frame_paths]
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=GIF_DURATION,
        loop=0,
    )
    print(f"  GIF → {gif_path}  ({len(imgs)} frames)")


def run():
    print("=" * 60)
    print("Experiment 08 — Pseudocode Simulations")
    print(f"Scenes : {', '.join(s for s, _, _ in SCENES)}")
    print(f"Frames : {N_STEPS} per scene")
    print("=" * 60)

    for scene_name, sim_cls, render_fn in SCENES:
        print(f"\n── {scene_name} ──")
        save_frames_and_gif(scene_name, sim_cls, render_fn)

    print("\n" + "=" * 60)
    print("EXP08 COMPLETE")
    print(f"Output in {RESULTS_ROOT}/")
    print("=" * 60)


if __name__ == "__main__":
    run()
