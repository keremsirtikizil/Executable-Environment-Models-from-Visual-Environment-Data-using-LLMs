"""
Experiment 08 — All PURE Pseudocode Simulations.

Converts every exp05/exp06 pseudocode directly into a runnable Python
simulation. Each simulation is a faithful implementation of the STATE
VARIABLES and IF/THEN RULES extracted by PURE — no LLM at runtime.

Scenes (9 total):
  bouncing_ball      exp05 CORRECT     — droplet on superhydrophobic surface
  newtons_cradle     exp05 PARTIAL     — 5-ball pendulum cradle, N-in/N-out
  pendulum           exp05 CORRECT     — spinning-wheel museum exhibit
  double_pendulum    exp05 —           — folding two-panel wall/door mechanism
  cymatics           exp05 PARTIAL     — Chladni sand patterns on vibrating plate
  metronomes         exp05 UNKNOWN     — 5 coupled oscillators, strong/weak coupling
  bowling_pins       exp06 PARTIAL     — full bowling game with pinsetter
  bowling_strike     exp06 PARTIAL     — strike physics with angular_velocity
  billiard_break     exp06 INCORRECT   — thermal-camera billiard shot with heat trails

Output:
  results/exp08/<scene>/simulation.gif   — animated GIF (180 frames each)
  results/exp08/combined.png             — one representative frame from all 9 scenes

Run from project root:
  /Users/kerem/.conda/envs/CS419_recit/bin/python3 experiments/exp08_all_simulations.py
"""

import os
import sys
import math
import glob
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

try:
    from PIL import Image
except ImportError:
    print("ERROR: pip install Pillow"); sys.exit(1)

RESULTS_ROOT = "results/exp08"
N_STEPS      = 180
GIF_MS       = 50
FIG_DPI      = 100

os.makedirs(RESULTS_ROOT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 1. BOUNCING BALL  (exp05 — CORRECT)
# Pseudocode: droplet.state ∈ {AIRBORNE, COMPRESSING, REBOUNDING}
# ═══════════════════════════════════════════════════════════════════
class BouncingBallSim:
    def __init__(self):
        self.surface_y      = 0.08
        self.initial_h      = 0.07
        self.min_h          = 0.025
        self.x              = 0.50
        self.y              = 0.82
        self.vy             = 0.0
        self.height         = self.initial_h
        self.width          = self.initial_h
        self.state          = "AIRBORNE"
        self.gravity        = 0.012
        self.deform_rate    = 0.006
        self._bounce        = 0
        self._impact_speed  = 0.0
        self._base_rest     = 0.55

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
            self.width  += self.deform_rate * 1.4
            if self.height <= self.min_h:
                self.state = "REBOUNDING"
        elif self.state == "REBOUNDING":
            self.height += self.deform_rate
            self.width  -= self.deform_rate * 1.4
            if self.height >= self.initial_h:
                self.height = self.initial_h
                self.width  = self.initial_h
                self.state  = "AIRBORNE"
                self.vy     = self._impact_speed * (self._base_rest ** (self._bounce + 1))
                self._bounce += 1
        if self._bounce > 7 or self.y > 1.05:
            self.__init__()

    def get_state(self):
        return dict(x=self.x, y=self.y, h=self.height, w=self.width,
                    state=self.state, bounce=self._bounce)

def render_bouncing_ball(state, ax, step):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a2e"); ax.axis("off")
    ax.axhline(y=0.08, color="#4ecdc4", lw=2.5)
    ax.text(0.5, 0.04, "superhydrophobic surface", ha="center",
            color="#4ecdc4", fontsize=7)
    col = {"AIRBORNE":"#74b9ff","COMPRESSING":"#fd79a8","REBOUNDING":"#55efc4"}[state["state"]]
    ax.add_patch(Ellipse((state["x"], state["y"]), width=state["w"],
                          height=state["h"], color=col, zorder=3))
    ax.set_title(f"Bouncing Ball · {state['state']} · bounce#{state['bounce']}",
                 color="white", fontsize=7, pad=3)
    ax.figure.patch.set_facecolor("#1a1a2e")


# ═══════════════════════════════════════════════════════════════════
# 2. NEWTON'S CRADLE  (exp05 — PARTIALLY CORRECT)
# Pseudocode: 5 balls, state ∈ {RESTING, SWINGING}, N-in/N-out rule
# ═══════════════════════════════════════════════════════════════════
class NewtonsCradleSim:
    N    = 5
    L    = 0.30          # string length (normalized)
    R    = 0.038
    GRAV = 0.0035
    DAMP = 0.0008

    def __init__(self):
        self.angles  = [0.0] * self.N
        self.omegas  = [0.0] * self.N
        self.state   = ["RESTING"] * self.N
        self._t      = 0
        self._phase  = 0  # cycles through demo scenarios
        self._start_scenario()

    def _start_scenario(self):
        n_pull = [1, 2, 3, 1][self._phase % 4]
        pull_angle = -0.9
        for i in range(self.N):
            self.angles[i] = 0.0
            self.omegas[i] = 0.0
            self.state[i]  = "RESTING"
        for i in range(n_pull):
            self.angles[i]    = pull_angle
            self.state[i]     = "SWINGING"

    def step(self):
        self._t += 1
        # pendulum physics on SWINGING balls
        for i in range(self.N):
            if self.state[i] == "SWINGING":
                self.omegas[i] += -self.GRAV * math.sin(self.angles[i])
                self.omegas[i] *= (1 - self.DAMP)
                self.angles[i] += self.omegas[i]

        # collision detection (right-moving swinging hits resting cluster)
        for i in range(self.N - 1):
            if (self.state[i] == "SWINGING" and self.omegas[i] > 0 and
                    abs(self.angles[i]) < 0.03 and self.state[i+1] == "RESTING"):
                # count swinging group on left
                g_start = i
                while g_start > 0 and self.state[g_start-1] == "SWINGING": g_start -= 1
                num_L = i - g_start + 1
                vel_in = self.omegas[i]
                # stop left group
                for k in range(g_start, i+1):
                    self.state[k] = "RESTING"
                    self.angles[k] = 0.0; self.omegas[k] = 0.0
                # launch equal number on right
                for k in range(self.N - num_L, self.N):
                    self.state[k] = "SWINGING"
                    self.omegas[k] = vel_in
                break

        # mirror: left-moving swinging hits resting cluster on left
        for i in range(self.N-1, 0, -1):
            if (self.state[i] == "SWINGING" and self.omegas[i] < 0 and
                    abs(self.angles[i]) < 0.03 and self.state[i-1] == "RESTING"):
                g_end = i
                while g_end < self.N-1 and self.state[g_end+1] == "SWINGING": g_end += 1
                num_R = g_end - i + 1
                vel_in = self.omegas[i]
                for k in range(i, g_end+1):
                    self.state[k] = "RESTING"
                    self.angles[k] = 0.0; self.omegas[k] = 0.0
                for k in range(0, num_R):
                    self.state[k] = "SWINGING"
                    self.omegas[k] = vel_in
                break

        if self._t > 160:
            self._t = 0; self._phase += 1; self._start_scenario()

    def get_state(self):
        return {"angles": list(self.angles), "states": list(self.state)}

def render_newtons_cradle(state, ax, step):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#0d0d1a"); ax.figure.patch.set_facecolor("#0d0d1a")
    cx_base = [0.30 + i * 0.10 for i in range(5)]
    pivot_y = 0.80
    L = 0.30
    colours = {"RESTING":"#b8c6db","SWINGING":"#f7c59f"}
    for i, (ang, st) in enumerate(zip(state["angles"], state["states"])):
        bx = cx_base[i] + L * math.sin(ang)
        by = pivot_y    - L * math.cos(ang)
        ax.plot([cx_base[i], bx], [pivot_y, by], color="#7f8c8d", lw=1.2)
        ax.add_patch(Circle((bx, by), 0.038, color=colours[st], zorder=3))
    # frame bar
    ax.plot([cx_base[0]-0.05, cx_base[4]+0.05], [pivot_y, pivot_y],
            color="#95a5a6", lw=3)
    ax.set_title("Newton's Cradle · N-in/N-out momentum transfer",
                 color="white", fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════
# 3. PENDULUM — spinning wheel / gyroscope exhibit  (exp05 — CORRECT)
# Pseudocode: mechanism_state, mechanism_rotation_speed (none→very_fast),
#             mechanism_configuration (extended/retracted), operator sequences
# ═══════════════════════════════════════════════════════════════════
_SPEEDS = ["none","very_slow","slow","fast","very_fast","slowing"]
_OPERATORS = ["boy","none","girl","none","woman","none","two_girls","none"]
_OP_TIMES  = [0, 40, 55, 90, 100, 130, 145, 170]

class PendulumGyroSim:
    def __init__(self):
        self.mech_state    = "stationary"
        self.speed_idx     = 0          # index into _SPEEDS
        self.config        = "extended"
        self.angle         = 0.0        # wheel rotation angle
        self.operator      = "none"
        self._t            = 0
        self._is_turning   = False

    def _get_operator(self):
        op = "none"
        for i, t in enumerate(_OP_TIMES):
            if self._t >= t:
                op = _OPERATORS[i]
        return op

    def step(self):
        self._t = (self._t + 1) % 180
        self.operator = self._get_operator()

        turn_windows = [(5,18),(60,80),(105,120),(150,165)]
        self._is_turning = any(a <= self._t <= b for a,b in turn_windows)

        if self._is_turning:
            self.mech_state = "rotating"
            if self.speed_idx < len(_SPEEDS) - 2:
                if self._t % 8 == 0:
                    self.speed_idx = min(self.speed_idx + 1, 4)
        else:
            if self.mech_state == "rotating":
                if self.speed_idx > 0:
                    if self._t % 12 == 0:
                        self.speed_idx = max(self.speed_idx - 1, 0)
                if self.speed_idx == 0:
                    self.mech_state = "stationary"

        # configuration: retracted at very_fast
        if self.speed_idx >= 4:
            self.config = "retracted"
        else:
            self.config = "extended"

        speed_vals = [0, 0.02, 0.06, 0.14, 0.25, 0.08]
        self.angle += speed_vals[self.speed_idx]

    def get_state(self):
        return dict(angle=self.angle, speed=_SPEEDS[self.speed_idx],
                    config=self.config, operator=self.operator,
                    turning=self._is_turning)

def render_pendulum_gyro(state, ax, step):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_facecolor("#1c1c1c"); ax.figure.patch.set_facecolor("#1c1c1c")
    cx, cy = 0.5, 0.5
    r_outer = 0.28 if state["config"]=="extended" else 0.20
    r_inner = 0.10
    # outer ring
    ring = Circle((cx,cy), r_outer, fill=False, edgecolor="#a0a0a0", lw=3)
    ax.add_patch(ring)
    # inner hub
    hub = Circle((cx,cy), r_inner, fill=False, edgecolor="#666", lw=2)
    ax.add_patch(hub)
    # spokes
    for k in range(8):
        ang = state["angle"] + k * math.pi / 4
        x1 = cx + r_inner * math.cos(ang)
        y1 = cy + r_inner * math.sin(ang)
        x2 = cx + r_outer * math.cos(ang)
        y2 = cy + r_outer * math.sin(ang)
        ax.plot([x1,x2],[y1,y2],color="#888",lw=1.5)
    # centre dot
    ax.add_patch(Circle((cx,cy),0.025,color="#e0e0e0"))
    # handle grip indicator
    if state["turning"]:
        ax.add_patch(Circle((cx+r_outer*0.7,cy-0.05),0.035,color="#f39c12",alpha=0.8))
        ax.text(cx+r_outer*0.7+0.05,cy-0.05,"← push",color="#f39c12",fontsize=6,va="center")

    col_map = {"none":"#555","very_slow":"#3498db","slow":"#2ecc71",
               "fast":"#f39c12","very_fast":"#e74c3c","slowing":"#9b59b6"}
    ax.set_title(f"Gyroscope · speed:{state['speed']} · {state['config']} · op:{state['operator']}",
                 color=col_map.get(state["speed"],"white"), fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════
# 4. DOUBLE PENDULUM — folding wall mechanism  (exp05 — wrong video)
# Pseudocode: System.state ∈ {LOCKED_WALL, ROTATABLE_DOOR, LOCKED_DOOR,
#             UNBOLTED_WALL, FOLDED_WALL}, PanelLeft/Right.isMobile
# ═══════════════════════════════════════════════════════════════════
_WALL_SEQUENCE = [
    ("LOCKED_WALL",    40),
    ("ROTATABLE_DOOR", 30),
    ("LOCKED_DOOR",    30),
    ("ROTATABLE_DOOR", 20),
    ("LOCKED_WALL",    20),
    ("UNBOLTED_WALL",  30),
    ("FOLDED_WALL",    40),
    ("UNBOLTED_WALL",  20),
]

class FoldingWallSim:
    def __init__(self):
        self._seq_idx  = 0
        self._hold     = 0
        self.sys_state = "LOCKED_WALL"
        self._anim     = 0.0   # 0→1 transition progress
        self._fold_ang = 0.0   # panel fold angle
        self._door_ang = 0.0   # door swing angle

    def step(self):
        self._hold += 1
        target_state, duration = _WALL_SEQUENCE[self._seq_idx]
        self._anim = min(1.0, self._hold / max(1, duration * 0.4))

        if self._hold >= duration:
            self._hold    = 0
            self._seq_idx = (self._seq_idx + 1) % len(_WALL_SEQUENCE)
            self.sys_state = _WALL_SEQUENCE[self._seq_idx][0]

        # animate angles
        if self.sys_state == "FOLDED_WALL":
            self._fold_ang = min(math.pi*0.45, self._fold_ang + 0.06)
        else:
            self._fold_ang = max(0.0, self._fold_ang - 0.06)

        if self.sys_state in ("ROTATABLE_DOOR", "LOCKED_DOOR"):
            self._door_ang = min(math.pi*0.55, self._door_ang + 0.05)
        else:
            self._door_ang = max(0.0, self._door_ang - 0.05)

    def get_state(self):
        mobile_L = self.sys_state == "FOLDED_WALL"
        mobile_R = self.sys_state in ("ROTATABLE_DOOR","LOCKED_DOOR","FOLDED_WALL")
        return dict(sys=self.sys_state, fold=self._fold_ang,
                    door=self._door_ang, mL=mobile_L, mR=mobile_R)

def render_folding_wall(state, ax, step):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_facecolor("#f5f0eb"); ax.figure.patch.set_facecolor("#f5f0eb")
    # hinge point
    hx, hy = 0.50, 0.50
    pw, ph = 0.35, 0.06
    # right panel (door swing)
    ra = state["door"]
    rx2 = hx + pw * math.cos(ra)
    ry2 = hy + pw * math.sin(ra - math.pi*0.5) * 0
    door_rect = plt.Polygon([
        [hx, hy-ph/2], [hx, hy+ph/2],
        [hx + pw*math.cos(ra), hy + pw*math.sin(ra)],
        [hx + pw*math.cos(ra), hy + pw*math.sin(ra) - ph]
    ], closed=True,
       facecolor="#d4a96a" if state["mR"] else "#8B7355",
       edgecolor="#5c4033", lw=1.5)
    ax.add_patch(door_rect)
    # left panel (fold)
    fa = math.pi - state["fold"]
    left_rect = plt.Polygon([
        [hx, hy-ph/2], [hx, hy+ph/2],
        [hx - pw*math.cos(state["fold"]), hy + pw*math.sin(state["fold"])],
        [hx - pw*math.cos(state["fold"]), hy + pw*math.sin(state["fold"]) - ph]
    ], closed=True,
       facecolor="#d4a96a" if state["mL"] else "#8B7355",
       edgecolor="#5c4033", lw=1.5)
    ax.add_patch(left_rect)
    # hinge bolt
    ax.add_patch(Circle((hx, hy), 0.025,
                          color="#e74c3c" if "LOCKED" in state["sys"] else "#2ecc71"))
    col_map = {"LOCKED_WALL":"#e74c3c","ROTATABLE_DOOR":"#3498db",
               "LOCKED_DOOR":"#9b59b6","UNBOLTED_WALL":"#f39c12","FOLDED_WALL":"#27ae60"}
    ax.set_title(f"Folding Wall · {state['sys']}",
                 color=col_map.get(state["sys"],"#333"), fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════
# 5. CYMATICS — Chladni patterns  (exp05 — PARTIAL, video had Hz labels)
# Pseudocode described intro transitions; simulate actual physics:
# sand accumulates on node lines of vibrating plate at resonant freq.
# ═══════════════════════════════════════════════════════════════════
_CHLADNI_MODES = [(1,2),(2,1),(2,3),(3,2),(1,4),(4,1),(3,4),(4,3),(2,5)]

class CymaticsSim:
    NPART = 600

    def __init__(self):
        self._mode_idx  = 0
        self._hold      = 0
        self._m, self._n = _CHLADNI_MODES[0]
        self.px = np.random.uniform(-1, 1, self.NPART)
        self.py = np.random.uniform(-1, 1, self.NPART)
        self.vx = np.random.uniform(-0.04,0.04,self.NPART)
        self.vy = np.random.uniform(-0.04,0.04,self.NPART)

    def _chladni(self, x, y):
        m, n = self._m, self._n
        return (np.cos(m*np.pi*x/2) * np.cos(n*np.pi*y/2) -
                np.cos(n*np.pi*x/2) * np.cos(m*np.pi*y/2))

    def step(self):
        self._hold += 1
        if self._hold > 120:
            self._hold    = 0
            self._mode_idx = (self._mode_idx + 1) % len(_CHLADNI_MODES)
            self._m, self._n = _CHLADNI_MODES[self._mode_idx]
            # scatter particles again for new mode
            self.px = np.random.uniform(-1, 1, self.NPART)
            self.py = np.random.uniform(-1, 1, self.NPART)
            self.vx = np.random.uniform(-0.04,0.04,self.NPART)
            self.vy = np.random.uniform(-0.04,0.04,self.NPART)

        # gradient of |chladni| → particles drift toward nodes (zeros)
        f    = self._chladni(self.px, self.py)
        eps  = 0.05
        gx   = (np.abs(self._chladni(self.px+eps, self.py)) -
                np.abs(self._chladni(self.px-eps, self.py))) / (2*eps)
        gy   = (np.abs(self._chladni(self.px, self.py+eps)) -
                np.abs(self._chladni(self.px, self.py-eps))) / (2*eps)
        drag = 0.88
        force = 0.008
        self.vx = self.vx * drag - gx * force
        self.vy = self.vy * drag - gy * force
        self.px = np.clip(self.px + self.vx, -1, 1)
        self.py = np.clip(self.py + self.vy, -1, 1)

    def get_state(self):
        return dict(px=self.px, py=self.py, m=self._m, n=self._n,
                    hold=self._hold)

def render_cymatics(state, ax, step):
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1)
    ax.set_facecolor("#0a0a0a"); ax.figure.patch.set_facecolor("#0a0a0a")
    ax.set_aspect("equal"); ax.axis("off")
    # plate border
    plate = patches.Rectangle((-1,-1),2,2,fill=False,edgecolor="#555",lw=2)
    ax.add_patch(plate)
    # particles
    ax.scatter(state["px"], state["py"], s=2, c="#f0e68c", alpha=0.6, linewidths=0)
    ax.set_title(f"Cymatics · Chladni mode ({state['m']},{state['n']}) · "
                 f"freq step {state['hold']}/120",
                 color="#f0e68c", fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════
# 6. METRONOMES  (exp05 — UNKNOWN)
# Pseudocode: 5 coupled oscillators on a movable board.
# Strong coupling (rollers): board moves → 2-vs-3 anti-phase attractor.
# Weak coupling (fixed board): chimera / alternating states.
# ═══════════════════════════════════════════════════════════════════
class MetronomesSim:
    N     = 5
    OMEGA = 0.18          # natural frequency
    DAMP  = 0.005

    def __init__(self):
        self._t      = 0
        self._phase  = 0   # 0=strong coupling, 1=weak coupling
        self._init_strong()

    def _init_strong(self):
        self._phase     = 0
        self.board_x    = 0.0
        self.board_v    = 0.0
        self.board_mass = 3.0    # light board → strong coupling
        self.angles  = [random.uniform(-1.0,1.0) for _ in range(self.N)]
        self.omegas  = [random.uniform(-0.05,0.05) for _ in range(self.N)]

    def _init_weak(self):
        self._phase     = 1
        self.board_x    = 0.0
        self.board_v    = 0.0
        self.board_mass = 50.0   # heavy board → weak coupling
        self.angles  = [random.uniform(-0.8,0.8) for _ in range(self.N)]
        self.omegas  = [random.uniform(-0.05,0.05) for _ in range(self.N)]

    def step(self):
        self._t += 1
        if self._t == 180:
            self._t = 0
            if self._phase == 0:
                self._init_weak()
            else:
                self._init_strong()

        # board acceleration from pendulum reactions
        board_force = sum(-math.sin(a) * self.OMEGA**2 for a in self.angles)
        self.board_v += board_force / self.board_mass
        self.board_v *= 0.995
        self.board_x += self.board_v

        # each metronome: pendulum + coupling through board acceleration
        board_acc = self.board_v * 0.3
        for i in range(self.N):
            torque = (-self.OMEGA**2 * math.sin(self.angles[i])
                      - self.DAMP * self.omegas[i]
                      - board_acc * math.cos(self.angles[i]))
            self.omegas[i] += torque
            self.angles[i] += self.omegas[i]
            self.angles[i]  = max(-1.2, min(1.2, self.angles[i]))

    def get_state(self):
        return dict(angles=list(self.angles), board_x=self.board_x,
                    phase=self._phase, t=self._t)

def render_metronomes(state, ax, step):
    ax.set_xlim(0,1); ax.set_ylim(0,0.9); ax.axis("off")
    bg = "#1a2a1a" if state["phase"]==0 else "#2a1a1a"
    ax.set_facecolor(bg); ax.figure.patch.set_facecolor(bg)

    bx_off = state["board_x"] * 0.04
    # board
    board = patches.Rectangle((0.05 + bx_off, 0.18), 0.90, 0.08,
                                facecolor="#8B6914", edgecolor="#5c4a00", lw=1.5)
    ax.add_patch(board)
    # rollers (strong coupling) or fixed (weak)
    for xi in [0.15, 0.50, 0.85]:
        col = "#aaaaaa" if state["phase"]==0 else "#444"
        ax.add_patch(Circle((xi + bx_off, 0.175), 0.025, color=col))

    # metronomes
    pos = [0.15 + i*0.18 for i in range(5)]
    for i, (a, px) in enumerate(zip(state["angles"], pos)):
        base_x = px + bx_off
        base_y = 0.26
        tip_x  = base_x + 0.12 * math.sin(a)
        tip_y  = base_y + 0.22
        ax.plot([base_x, tip_x], [base_y, tip_y], color="#ddd", lw=2)
        ax.add_patch(Circle((tip_x, tip_y), 0.018, color="#e74c3c", zorder=4))
        ax.add_patch(Circle((base_x, base_y), 0.012, color="#666"))

    label = "Strong coupling (rollers)" if state["phase"]==0 else "Weak coupling (fixed)"
    ax.set_title(f"Metronomes · {label} · t={state['t']}",
                 color="white", fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════
# 7. BOWLING PINS  (exp06 — PARTIALLY CORRECT)
# Full game loop with pinsetter, cascade, spare/strike logic
# ═══════════════════════════════════════════════════════════════════
PIN_SP = 0.075; BALL_R = 0.038; PIN_R = 0.022
BALL_SPD = 0.018; COL_D = BALL_R + PIN_R; PIN_COL_D = PIN_R * 2.2

def _pin_init_positions():
    d = PIN_SP
    return [(0.500,0.740),(0.500-d*.5,0.740+d*.87),(0.500+d*.5,0.740+d*.87),
            (0.500-d,0.740+d*1.74),(0.500,0.740+d*1.74),(0.500+d,0.740+d*1.74),
            (0.500-d*1.5,0.740+d*2.61),(0.500-d*.5,0.740+d*2.61),
            (0.500+d*.5,0.740+d*2.61),(0.500+d*1.5,0.740+d*2.61)]

class BowlingPinsSim:
    def __init__(self): self._init(); self._frame=0

    def _init(self):
        self.pins=[{"x":x,"y":y,"vx":0.,"vy":0.,"state":"standing","ix":x,"iy":y}
                   for x,y in _pin_init_positions()]
        self.bx=0.5; self.by=0.08; self.bvx=0.; self.bvy=BALL_SPD
        self.bvis=True; self.gs="ready"; self.rc="idle"
        self.roll=1; self.hit_type="none"; self.ui=""

    def _dist(self,ax,ay,bx,by): return math.hypot(ax-bx,ay-by)

    def step(self):
        self._frame+=1
        if self.gs=="ready":
            if self._frame%55==1:
                self.bx=0.5+random.uniform(-0.03,0.03)
                self.by=0.08; self.bvy=BALL_SPD
                self.bvx=(0.5-self.bx)*0.05
                self.bvis=True; self.gs="ball_rolling"; self.hit_type="none"; self.ui=""
        elif self.gs=="ball_rolling":
            self.bx+=self.bvx; self.by+=self.bvy
            for p in self.pins:
                if p["state"]=="standing" and self._dist(self.bx,self.by,p["x"],p["y"])<COL_D:
                    if self.hit_type=="none":
                        dx=self.bx-p["x"]
                        self.hit_type="brooklyn" if dx<-0.01 else "pocket" if dx>0.01 else "nose_hit"
                    p["state"]="falling"
                    ang=math.atan2(p["y"]-self.by,p["x"]-self.bx)
                    spd=0.022+random.uniform(0,.01)
                    p["vx"]=math.cos(ang)*spd; p["vy"]=math.sin(ang)*spd
                    self.bvx*=0.85; self.gs="pins_interacting"
            if self.by>1.05: self.bvis=False; self.gs="pins_interacting"
        elif self.gs=="pins_interacting":
            if self.bvis:
                self.bx+=self.bvx; self.by+=self.bvy
                if self.by>1.05: self.bvis=False
            moving=False
            for p in self.pins:
                if p["state"]=="falling":
                    moving=True
                    p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vx"]*=0.91; p["vy"]*=0.91
                    if abs(p["vx"])<0.0008 and abs(p["vy"])<0.0008: p["state"]="fallen"
                    for q in self.pins:
                        if q["state"]=="standing" and q is not p:
                            if self._dist(p["x"],p["y"],q["x"],q["y"])<PIN_COL_D:
                                q["state"]="falling"
                                ang=math.atan2(q["y"]-p["y"],q["x"]-p["x"])
                                spd=math.hypot(p["vx"],p["vy"])*0.65
                                q["vx"]=math.cos(ang)*spd; q["vy"]=math.sin(ang)*spd
            if not moving and not self.bvis:
                self.gs="end_of_roll"; self._rt=0
        elif self.gs=="end_of_roll":
            self._rt=getattr(self,"_rt",0)+1
            if self._rt==1:
                st=[p for p in self.pins if p["state"]=="standing"]
                if self.roll==1:
                    self.ui="Strike!" if not st else f"Roll 1: {10-len(st)} down"
                    self.rc="resetting_frame" if not st else "preparing_spare"
                else:
                    self.ui=f"Spare!" if not st else f"Roll 2: {10-len(st)} total"
                    self.rc="resetting_frame"
            if self._rt>45: self._do_reset()

    def _do_reset(self):
        if self.rc=="preparing_spare":
            for p in self.pins:
                if p["state"] in ("fallen","falling"): p["state"]="cleared"
            self.roll=2
        else:
            self._init(); return
        self.rc="idle"; self.gs="ready"; self._frame=0

    def get_state(self):
        return {"bx":self.bx,"by":self.by,"bvis":self.bvis,
                "pins":[(p["x"],p["y"],p["state"]) for p in self.pins],
                "gs":self.gs,"ui":self.ui,"ht":self.hit_type}

def render_bowling_pins(state, ax, step):
    ax.set_xlim(0.2,0.8); ax.set_ylim(0.0,1.05); ax.axis("off")
    ax.set_facecolor("#2d1b00"); ax.figure.patch.set_facecolor("#2d1b00")
    ax.add_patch(patches.Rectangle((0.30,0.00),0.40,1.05,lw=1.5,
                  edgecolor="#c8a96e",facecolor="#c8a96e22"))
    ax.axhline(y=0.12,color="#ff6b6b",lw=1.5,xmin=0.25,xmax=0.75)
    if state["bvis"]:
        ax.add_patch(Circle((state["bx"],state["by"]),BALL_R,color="#2c3e50",zorder=4))
        ax.add_patch(Circle((state["bx"],state["by"]),BALL_R,fill=False,
                              edgecolor="#95a5a6",lw=1.5,zorder=5))
    pc={"standing":"#ecf0f1","falling":"#e74c3c","fallen":"#7f8c8d","cleared":None}
    for px,py,ps in state["pins"]:
        c=pc.get(ps)
        if c: ax.add_patch(Circle((px,py),PIN_R,color=c,zorder=3))
    if state["ui"]:
        ax.text(0.5,0.55,state["ui"],ha="center",color="#f1c40f",fontsize=14,
                fontweight="bold",bbox=dict(boxstyle="round",facecolor="#00000088"),zorder=6)
    ax.set_title(f"Bowling · {state['gs']} · {state['ht']}",color="white",fontsize=7,pad=3)


# ═══════════════════════════════════════════════════════════════════
# 8. BOWLING STRIKE  (exp06 — PARTIALLY CORRECT)
# Pseudocode: pin.angular_velocity, calculate_imparted_velocity,
#             score_display.text, COLLISION_DISTANCE, PIN_DECK_END_Z
# ═══════════════════════════════════════════════════════════════════
class BowlingStrikeSim:
    NUM_PINS         = 10
    COLLISION_DIST   = COL_D
    PIN_DECK_END_Z   = 1.05

    def __init__(self):
        self._reset(); self._t=0

    def _reset(self):
        self.pins=[{"x":x,"y":y,"vx":0.,"vy":0.,"ang":0.,"ang_v":0.,
                    "state":"standing","ix":x,"iy":y}
                   for x,y in _pin_init_positions()]
        self.ball={"x":0.5,"y":0.08,"vx":0.,"vy":BALL_SPD,"visible":True}
        self.game_state="AWAITING_ROLL"
        self.roll_number=1; self.first_roll_score=0
        self.score_text=""; self.pins_settled=False

    def step(self):
        self._t+=1
        if self.game_state=="AWAITING_ROLL":
            if self._t%55==1:
                self.ball["x"]=0.5+random.uniform(-0.02,0.02)
                self.ball["y"]=0.08; self.ball["vy"]=BALL_SPD; self.ball["vx"]=0.
                self.ball["visible"]=True; self.game_state="BALL_IN_PLAY"
                self.pins_settled=False

        elif self.game_state=="BALL_IN_PLAY":
            b=self.ball
            b["x"]+=b["vx"]; b["y"]+=b["vy"]
            if b["y"]>self.PIN_DECK_END_Z:
                b["visible"]=False; self.game_state="POST_ROLL"
            else:
                for p in self.pins:
                    if p["state"]=="standing" and math.hypot(b["x"]-p["x"],b["y"]-p["y"])<self.COLLISION_DIST:
                        p["state"]="falling"
                        ang=math.atan2(p["y"]-b["y"],p["x"]-b["x"])
                        spd=0.024+random.uniform(0,.008)
                        p["vx"]=math.cos(ang)*spd; p["vy"]=math.sin(ang)*spd
                        p["ang_v"]=(random.random()-0.5)*0.3  # angular velocity
                        b["vx"]*=0.85

        if self.game_state in ("BALL_IN_PLAY","POST_ROLL"):
            any_mv=False
            for p in self.pins:
                if p["state"]=="falling":
                    any_mv=True
                    p["x"]+=p["vx"]; p["y"]+=p["vy"]
                    p["vx"]*=0.91; p["vy"]*=0.91
                    p["ang"]+=p["ang_v"]; p["ang_v"]*=0.93
                    if abs(p["vx"])<0.001 and abs(p["vy"])<0.001: p["state"]="fallen"
                    for q in self.pins:
                        if q is not p and q["state"]=="standing":
                            if math.hypot(p["x"]-q["x"],p["y"]-q["y"])<self.COLLISION_DIST:
                                ang=math.atan2(q["y"]-p["y"],q["x"]-p["x"])
                                spd=math.hypot(p["vx"],p["vy"])*0.65
                                q["state"]="falling"; q["vx"]=math.cos(ang)*spd
                                q["vy"]=math.sin(ang)*spd; q["ang_v"]=(random.random()-0.5)*0.25
            if self.game_state=="POST_ROLL" and not any_mv:
                self.pins_settled=True

        if self.game_state=="POST_ROLL" and self.pins_settled:
            fallen=[p for p in self.pins if p["state"] in ("fallen","falling")]
            n=len(fallen)
            if self.roll_number==1:
                self.first_roll_score=n; self.score_text=str(n)
                if n==self.NUM_PINS:
                    self.score_text="STRIKE! ★"
                    self._schedule_reset(50)
                else:
                    self._schedule_reset(40)
                    self.roll_number=2
            else:
                total=self.first_roll_score+n
                self.score_text=f"SPARE! {total}" if n==(10-self.first_roll_score) else str(total)
                self._schedule_reset(50)
            self.game_state="DONE"

        if self.game_state=="DONE":
            self._rt=getattr(self,"_rt",0)+1
            if self._rt>=getattr(self,"_rt_max",50): self._reset()

    def _schedule_reset(self,delay):
        self._rt=0; self._rt_max=delay

    def get_state(self):
        return {"ball":self.ball,"pins":self.pins,
                "gs":self.game_state,"score":self.score_text}

def render_bowling_strike(state, ax, step):
    ax.set_xlim(0.2,0.8); ax.set_ylim(0.0,1.05); ax.axis("off")
    ax.set_facecolor("#1a2d00"); ax.figure.patch.set_facecolor("#1a2d00")
    ax.add_patch(patches.Rectangle((0.30,0.00),0.40,1.05,lw=1.5,
                  edgecolor="#90c8a0",facecolor="#90c8a022"))
    ax.axhline(y=0.12,color="#ff6b6b",lw=1.5,xmin=0.25,xmax=0.75)
    b=state["ball"]
    if b["visible"]:
        ax.add_patch(Circle((b["x"],b["y"]),BALL_R,color="#1a1a3e",zorder=4))
        ax.add_patch(Circle((b["x"],b["y"]),BALL_R,fill=False,edgecolor="#aaa",lw=1.5,zorder=5))
    for p in state["pins"]:
        c={"standing":"#dfe6e9","falling":"#fd79a8","fallen":"#636e72"}.get(p["state"])
        if c:
            ax.add_patch(Circle((p["x"],p["y"]),PIN_R,color=c,zorder=3))
            # show angular velocity as a rotation indicator line
            if p["state"]=="falling" and abs(p.get("ang_v",0))>0.02:
                a=p.get("ang",0)
                ax.plot([p["x"],p["x"]+PIN_R*1.8*math.cos(a)],
                        [p["y"],p["y"]+PIN_R*1.8*math.sin(a)],
                        color="#fdcb6e",lw=1.2,zorder=4)
    if state["score"]:
        ax.text(0.5,0.55,state["score"],ha="center",color="#fdcb6e",fontsize=13,
                fontweight="bold",bbox=dict(boxstyle="round",facecolor="#00000088"),zorder=6)
    ax.set_title(f"Bowling Strike · {state['gs']}",color="white",fontsize=7,pad=3)


# ═══════════════════════════════════════════════════════════════════
# 9. BILLIARD BREAK — THERMAL VIEW  (exp06 — INCORRECT verdict but
#    correctly described thermal camera physics)
# Pseudocode: viewMode NORMAL/THERMAL, cueStick.temperature,
#             ball.temperature, heatTrail, mistEffect, table.color
# ═══════════════════════════════════════════════════════════════════
THERM_CMAP = LinearSegmentedColormap.from_list(
    "thermal", ["#000080","#000080","#0000ff","#00ffff",
                "#00ff00","#ffff00","#ff6600","#ff0000","#ffffff"])

class BilliardThermalSim:
    TABLE_W = 1.0; TABLE_H = 0.55; BR = 0.022; FRICTION = 0.987

    def __init__(self): self._reset()

    def _reset(self):
        self._t         = 0
        self.view_mode  = "NORMAL"
        self.table_col  = "#1a6b3a"
        self.has_struck = False
        # cue stick
        self.cue = {"x":0.05,"y":self.TABLE_H/2,"vx":0.06,"vis":True,"temp":20.0}
        # three balls
        cx = 0.70; cy = self.TABLE_H/2
        self.balls = [
            {"x":0.22,"y":cy,"vx":0.,"vy":0.,"temp":20.,"vis":True,"col":"#ffffff","name":"white"},
            {"x":cx,  "y":cy,"vx":0.,"vy":0.,"temp":20.,"vis":True,"col":"#f1c40f","name":"yellow"},
            {"x":cx+self.BR*2.2,"y":cy+0.04,"vx":0.,"vy":0.,"temp":20.,"vis":True,"col":"#c0392b","name":"spotted"},
        ]
        self.heat_trail = []   # list of (x,y,temp,life)
        self.mist = {"x":0.,"y":0.,"vis":False,"life":0}
        self.spotlight = 1.0

    def _collide(self, a, b):
        dx=b["x"]-a["x"]; dy=b["y"]-a["y"]; d=math.hypot(dx,dy)
        if d<self.BR*2 and d>1e-9:
            nx,ny=dx/d,dy/d
            ov=self.BR*2-d; a["x"]-=nx*ov/2; b["x"]+=nx*ov/2
            a["y"]-=ny*ov/2; b["y"]+=ny*ov/2
            dvn=(b["vx"]-a["vx"])*nx+(b["vy"]-a["vy"])*ny
            if dvn<0:
                a["vx"]+=dvn*nx; a["vy"]+=dvn*ny
                b["vx"]-=dvn*nx; b["vy"]-=dvn*ny
                # heat on collision
                heat=abs(dvn)*300
                a["temp"]=min(100,a["temp"]+heat*0.6)
                b["temp"]=min(100,b["temp"]+heat*0.6)
                # mist effect
                self.mist={"x":(a["x"]+b["x"])/2,"y":(a["y"]+b["y"])/2,
                           "vis":True,"life":15}
            return True
        return False

    def step(self):
        self._t+=1

        # cue strike sequence
        if self._t==15 and not self.has_struck:
            self.has_struck=True
            self.balls[0]["vx"]=0.038
            self.cue["temp"]=75.
            self.balls[0]["temp"]=55.

        if self._t==20: self.view_mode="THERMAL"; self.table_col="#003333"
        if self._t==23: self.view_mode="NORMAL";  self.table_col="#1a6b3a"

        # thermal cooldown
        for b in self.balls:
            b["temp"]=max(20.,b["temp"]-0.8)
        self.cue["temp"]=max(20.,self.cue["temp"]-1.5)

        # move balls + bounce
        for b in self.balls:
            if not b["vis"]: continue
            b["x"]+=b["vx"]; b["y"]+=b["vy"]
            b["vx"]*=self.FRICTION; b["vy"]*=self.FRICTION
            if b["x"]-self.BR<0:     b["x"]=self.BR;          b["vx"]=abs(b["vx"]); b["temp"]=min(100,b["temp"]+5)
            if b["x"]+self.BR>self.TABLE_W: b["x"]=self.TABLE_W-self.BR; b["vx"]=-abs(b["vx"]); b["temp"]=min(100,b["temp"]+5)
            if b["y"]-self.BR<0:     b["y"]=self.BR;          b["vy"]=abs(b["vy"]); b["temp"]=min(100,b["temp"]+5)
            if b["y"]+self.BR>self.TABLE_H: b["y"]=self.TABLE_H-self.BR; b["vy"]=-abs(b["vy"]); b["temp"]=min(100,b["temp"]+5)
            # add to heat trail
            if b["temp"]>30:
                self.heat_trail.append({"x":b["x"],"y":b["y"],
                                        "temp":b["temp"],"life":25})
        # ball-ball collisions
        for i in range(len(self.balls)):
            for j in range(i+1,len(self.balls)):
                self._collide(self.balls[i],self.balls[j])
        # age heat trail
        self.heat_trail=[h for h in self.heat_trail if h["life"]>0]
        for h in self.heat_trail: h["life"]-=1; h["temp"]*=0.96
        # mist
        if self.mist["vis"]:
            self.mist["life"]-=1
            if self.mist["life"]<=0: self.mist["vis"]=False

        if self._t>160: self._reset()

    def get_state(self):
        return dict(t=self._t,view=self.view_mode,table=self.table_col,
                    balls=self.balls,trail=list(self.heat_trail),
                    mist=dict(self.mist),cue=dict(self.cue))

def render_billiard_thermal(state, ax, step):
    vm = state["view"]
    bg = "#001a1a" if vm=="THERMAL" else "#1a0a00"
    ax.set_facecolor(bg); ax.figure.patch.set_facecolor(bg)
    ax.set_xlim(-0.02,state["table"]+0.02 if isinstance(state["table"],float) else 1.02)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,0.57); ax.axis("off")
    tc = state["table"]
    ax.add_patch(patches.Rectangle((0,0),1.0,0.55,lw=3,
                  edgecolor="#8B4513",facecolor=tc))
    # heat trail
    for h in state["trail"]:
        t_norm=(h["temp"]-20)/80.; alpha=h["life"]/25.*0.6
        col=THERM_CMAP(max(0.,min(1.,t_norm))) if vm=="THERMAL" else "#ff660022"
        ax.add_patch(Circle((h["x"],h["y"]),0.012,color=col,alpha=alpha,zorder=2))
    # balls
    for b in state["balls"]:
        if not b["vis"]: continue
        if vm=="THERMAL":
            t_n=(b["temp"]-20)/80.
            c=THERM_CMAP(max(0.,min(1.,t_n)))
        else:
            c=b["col"]
        ax.add_patch(Circle((b["x"],b["y"]),0.022,color=c,zorder=4))
        if vm=="NORMAL":
            ax.add_patch(Circle((b["x"],b["y"]),0.022,fill=False,
                                  edgecolor="#ffffff44",lw=0.8,zorder=5))
    # mist
    m=state["mist"]
    if m["vis"]:
        ax.add_patch(Circle((m["x"],m["y"]),0.04+0.01*(15-m["life"]),
                              color="#ffffff",alpha=m["life"]/20.,zorder=6))
    ax.set_title(f"Billiard Thermal · viewMode:{state['view']} · t={state['t']}",
                 color="#00ffff" if vm=="THERMAL" else "white", fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════
SIMULATIONS = [
    ("bouncing_ball",      BouncingBallSim,     render_bouncing_ball),
    ("newtons_cradle",     NewtonsCradleSim,    render_newtons_cradle),
    ("pendulum_gyro",      PendulumGyroSim,     render_pendulum_gyro),
    ("double_pendulum_wall", FoldingWallSim,    render_folding_wall),
    ("cymatics",           CymaticsSim,         render_cymatics),
    ("metronomes",         MetronomesSim,       render_metronomes),
    ("bowling_pins",       BowlingPinsSim,      render_bowling_pins),
    ("bowling_strike",     BowlingStrikeSim,    render_bowling_strike),
    ("billiard_thermal",   BilliardThermalSim,  render_billiard_thermal),
]


def save_gif(scene_name, sim_cls, render_fn):
    out_dir = os.path.join(RESULTS_ROOT, scene_name)
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "simulation.gif")
    if os.path.isfile(gif_path):
        print(f"  [SKIP] {scene_name} — GIF already exists"); return gif_path

    sim = sim_cls()
    frames = []
    print(f"  Rendering {N_STEPS} frames for {scene_name}...")
    for i in range(N_STEPS):
        sim.step()
        state = sim.get_state()
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=FIG_DPI)
        render_fn(state, ax, i)
        plt.tight_layout(pad=0.4)
        png = os.path.join(out_dir, f"frame_{i:03d}.png")
        fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        frames.append(png)
        if (i+1) % 60 == 0: print(f"    {i+1}/{N_STEPS}")

    imgs = [Image.open(p) for p in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=GIF_MS, loop=0)
    print(f"  GIF → {gif_path}")
    return gif_path


def make_combined(representative_step=60):
    """One representative frame from each simulation in a 3×3 grid."""
    print("\nBuilding combined overview image...")
    n = len(SIMULATIONS)
    cols, rows = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 13.5), dpi=80)
    fig.patch.set_facecolor("#111111")
    fig.suptitle("PURE — Pseudocode Simulations (all 9 scenes)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    for idx, (name, sim_cls, render_fn) in enumerate(SIMULATIONS):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sim = sim_cls()
        for _ in range(representative_step):
            sim.step()
        state = sim.get_state()
        render_fn(state, ax, representative_step)

    plt.tight_layout(pad=0.5)
    out = os.path.join(RESULTS_ROOT, "combined.png")
    fig.savefig(out, dpi=80, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"Combined → {out}")


def run():
    print("=" * 60)
    print("Experiment 08 — All PURE Pseudocode Simulations")
    print(f"Scenes : {len(SIMULATIONS)}  |  Frames/scene : {N_STEPS}")
    print("=" * 60)
    for name, sim_cls, render_fn in SIMULATIONS:
        print(f"\n── {name} ──")
        save_gif(name, sim_cls, render_fn)
    make_combined()
    print("\n" + "=" * 60)
    print("DONE  →  results/exp08/")
    print("=" * 60)


if __name__ == "__main__":
    run()
