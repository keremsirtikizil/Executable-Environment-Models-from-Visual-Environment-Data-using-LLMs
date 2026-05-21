import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, Ellipse
from PIL import Image

# --- Constants and Configuration ---
OUTPUT_DIR = "results/exp08/billiards_physics"
NUM_FRAMES = 120
FRAME_DURATION_MS = 60
FIG_SIZE = (8, 6)
DPI = 100

# Table and Ball dimensions
TABLE_WIDTH = 4.0
TABLE_HEIGHT = 2.0
BALL_RADIUS = 0.05
POCKET_RADIUS = 0.1
SPEED = 0.05  # units per frame

# --- Ball Object ---
class Ball:
    def __init__(self, pos, color='white', radius=BALL_RADIUS):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.angle = 0.0
        self.angular_vel = 0.0
        self.color = color
        self.radius = radius
        self.visible = True
        self.trail = []

    def update(self):
        self.pos += self.vel
        self.angle += self.angular_vel
        if np.any(self.vel):
            self.trail.append(self.pos.copy())
            if len(self.trail) > 20:
                self.trail.pop(0)

# --- Simulation State ---
class SimulationState:
    def __init__(self):
        # Scene and timing
        self.scene = "IntroTitle"
        self.scene_timer = 0

        # Ball objects
        self.balls = {
            'cue': Ball([2.0, 0.5], 'white'),
            'obj1': Ball([2.0, 1.2], 'yellow'),
            'striped': Ball([2.0, 1.0], 'purple'),
            'analysis': Ball([3.5, 1.8], 'red'),
            'masse': Ball([1.0, 0.5], 'white'),
            'slow_mo': Ball([2.0, 1.0], 'white'),
            'throw_obj': Ball([1.5, 1.0], 'blue'),
            'hop': Ball([1.5, 1.0], 'white'),
            'stun_obj': Ball([2.0, 1.0], 'orange'),
            'push_obj': Ball([2.0, 1.05 + BALL_RADIUS], 'red'),
            'legal_obj': Ball([2.5, 1.5], 'green'),
            'credit_obj': Ball([3.0, 1.0], 'yellow'),
        }
        self.cue_stick_pos = np.array([0.0, 0.0])
        self.cue_stick_angle = 0.0
        self.cue_stick_visible = False
        self.chalk_dust = []

        # Boolean state flags from pseudocode
        flags = [
            "cue_ball_moving", "object_ball_moving", "striped_ball_spinning",
            "analysis_ball_moving", "masse_ball_moving", "draw_shot_cue_ball_moving",
            "slow_mo_ball_jumping", "cue_tip_moving", "throw_demo_cue_ball_moving",
            "throw_demo_object_ball_moving", "hop_analysis_ball_moving",
            "stun_shot_cue_ball_moving", "stun_shot_object_ball_moving",
            "push_shot_cue_ball_moving", "push_shot_object_ball_moving",
            "legal_shot_cue_ball_moving", "legal_shot_object_ball_moving",
            "final_shot_cue_ball_moving", "wrap_up_cue_tip_moving",
            "credit_shot_cue_ball_moving", "display_intro_title",
            "display_camera_info", "display_rattle_title", "display_analysis_graphics",
            "display_curved_path_title", "display_masse_shot_graphics",
            "display_thermal_highlight", "display_thermal_grooves_graphics",
            "display_draw_shot_diagram", "display_slow_mo_info",
            "display_draw_shot_spin_axis", "display_squirt_title",
            "display_squirt_analysis_graphics", "display_squirt_cause_text",
            "display_squirt_line_graphic", "display_throw_title",
            "display_throw_analysis_graphics", "display_hop_stun_title",
            "display_hop_analysis_graphics", "display_push_shot_title",
            "display_push_shot_analysis_graphics", "display_wrap_up_title",
            "display_credit_watermark"
        ]
        for flag in flags:
            setattr(self, flag, False)

        # Other state variables
        self.aiming_diagram_state = "CenterDraw"
        self.push_shot_phase = "Setup"
        self.legal_shot_phase = "Setup"
        self.video_filter = "None"
        self.deform_ball = False

    def reset_ball_states(self):
        for ball in self.balls.values():
            ball.vel = np.array([0.0, 0.0])
            ball.angular_vel = 0.0
            ball.trail = []
        
        moving_flags = [k for k in self.__dict__.keys() if 'moving' in k]
        for flag in moving_flags:
            setattr(self, flag, False)

# --- Physics and State Update ---
def update_state(state, frame_num):
    state.scene_timer = frame_num // 3  # Pacing to match pseudocode timers

    # --- State Transitions and Rule Logic ---
    prev_scene = state.scene

    if state.scene == "IntroTitle":
        state.display_intro_title = True
        if state.scene_timer > 1:
            state.scene = "IntroShot"
            state.display_intro_title = False
            state.cue_ball_moving = True
            cue = state.balls['cue']
            obj = state.balls['obj1']
            direction = obj.pos - cue.pos
            cue.vel = direction / np.linalg.norm(direction) * SPEED

    elif state.scene == "IntroShot":
        if state.cue_ball_moving:
            cue = state.balls['cue']
            obj = state.balls['obj1']
            if np.linalg.norm(cue.pos - obj.pos) < cue.radius + obj.radius:
                state.cue_ball_moving = False
                state.object_ball_moving = True
                cue.vel = np.array([0.0, 0.0])
                obj.vel = np.array([0.0, SPEED * 0.8]) # Towards pocket
        if state.scene_timer > 3:
            state.scene = "SpinDemo"
            state.striped_ball_spinning = True
            state.display_camera_info = True

    elif state.scene == "SpinDemo":
        if state.striped_ball_spinning:
            state.balls['striped'].angular_vel = 0.5
        if state.scene_timer > 4:
            state.scene = "RattleSetup"
            state.striped_ball_spinning = False
            state.display_camera_info = False

    elif state.scene == "RattleSetup" and state.scene_timer > 5:
        state.scene = "RattleTitle"
        state.display_rattle_title = True

    elif state.scene == "RattleTitle" and state.scene_timer > 6:
        state.scene = "RattleAnalysis"
        state.display_rattle_title = False
        state.display_analysis_graphics = True
        state.analysis_ball_moving = True
        state.balls['analysis'].vel = np.array([SPEED, -SPEED*0.2])

    elif state.scene == "RattleAnalysis":
        state.video_filter = "BlackAndWhite"
        ball = state.balls['analysis']
        if state.analysis_ball_moving and ball.pos[0] > TABLE_WIDTH - 0.2:
             ball.vel[0] *= -1 # Bounce off pocket wall
             ball.vel[1] *= 1.2
        if state.scene_timer > 7:
            state.scene = "CurvedPathTitle"
            state.display_analysis_graphics = False
            state.analysis_ball_moving = False
            state.video_filter = "None"
            state.display_curved_path_title = True

    elif state.scene == "CurvedPathTitle":
        if state.scene_timer > 8:
            state.scene = "MasseShotDemo"
            state.display_curved_path_title = False
            state.display_masse_shot_graphics = True
            state.masse_ball_moving = True
            state.balls['masse'].vel = np.array([SPEED * 0.8, SPEED * 1.2])

    elif state.scene == "MasseShotDemo":
        if state.masse_ball_moving:
            # Simulate curve by altering velocity
            vel = state.balls['masse'].vel
            vel[0] -= 0.0015
            vel[1] = max(0, vel[1] - 0.002)
        if state.scene_timer > 9:
            state.scene = "ThermalAnalysis"
            state.display_masse_shot_graphics = False
            state.masse_ball_moving = False

    elif state.scene == "ThermalAnalysis":
        if state.scene_timer > 10:
            state.display_thermal_highlight = True
        if state.scene_timer > 11:
            state.scene = "ThermalGroovesAnalysis"
            state.display_thermal_highlight = False
            state.display_thermal_grooves_graphics = True

    elif state.scene == "ThermalGroovesAnalysis":
        if state.scene_timer > 12:
            state.scene = "DrawShotSetup"
            state.display_thermal_grooves_graphics = False
            state.display_draw_shot_diagram = True
            state.aiming_diagram_state = "CenterDraw"

    elif state.scene == "DrawShotSetup":
        if state.scene_timer > 13:
            state.scene = "DrawShotExecution"
            state.display_draw_shot_diagram = False
            state.draw_shot_cue_ball_moving = True
            cue = state.balls['cue']
            obj = state.balls['obj1']
            cue.pos = np.array([2.0, 0.5])
            direction = obj.pos - cue.pos
            cue.vel = direction / np.linalg.norm(direction) * SPEED

    elif state.scene == "DrawShotExecution":
        if state.draw_shot_cue_ball_moving:
            cue = state.balls['cue']
            obj = state.balls['obj1']
            if np.linalg.norm(cue.pos - obj.pos) < cue.radius + obj.radius:
                state.draw_shot_cue_ball_moving = False
                cue.vel *= -0.5 # Simple draw back effect
        if state.scene_timer > 14:
            state.scene = "DrawShotSlowMo"
            state.display_slow_mo_info = True
            state.slow_mo_ball_jumping = True

    elif state.scene == "DrawShotSlowMo":
        state.video_filter = "BlackAndWhite"
        if state.slow_mo_ball_jumping:
            ball = state.balls['slow_mo']
            if ball.vel[1] == 0: ball.vel = np.array([0.0, 0.02]) # Initial hop
            ball.vel[1] -= 0.001 # Gravity
        if state.scene_timer > 15:
            state.display_draw_shot_spin_axis = True
        if state.scene_timer > 16:
            state.scene = "SquirtTitle"
            state.display_slow_mo_info = False
            state.display_draw_shot_spin_axis = False
            state.video_filter = "None"
            state.display_squirt_title = True
            state.aiming_diagram_state = "RightEnglish"

    elif state.scene == "SquirtTitle" and state.scene_timer > 17:
        state.scene = "SquirtDemoSetup"
        state.display_squirt_title = False

    elif state.scene == "SquirtDemoSetup" and state.scene_timer > 18:
        state.scene = "SquirtDemoAnalysis"
        state.display_squirt_analysis_graphics = True

    elif state.scene == "SquirtDemoAnalysis" and state.scene_timer > 19:
        state.scene = "SquirtCauseAnalysis"
        state.display_squirt_analysis_graphics = False
        state.video_filter = "Black