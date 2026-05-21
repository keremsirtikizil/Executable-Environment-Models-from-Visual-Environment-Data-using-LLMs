import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def run_simulation():
    """
    Implements a physics simulation of a bouncing droplet based on pseudocode,
    renders each step to a PNG file, and compiles the frames into an animated GIF.
    """
    # --- Configuration and Constants ---
    OUTPUT_DIR = "results/exp08/bouncing_ball"
    N_FRAMES = 120
    FIG_SIZE = (8, 6)
    DPI = 100
    GIF_DURATION = 60  # ms per frame

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- STATE VARIABLES Initialization ---
    droplet = {
        'y': 350.0,
        'velocity_y': 0.0,
        'height': 30.0,
        'width': 30.0,
        'initial_height': 30.0,
        'state': 'AIRBORNE',  # [AIRBORNE, COMPRESSING, REBOUNDING]
        'min_height': 15.0,
    }

    surface_y = 50.0
    gravity = 0.5
    deformation_rate = 1.2
    restitution_velocity = 9.0

    # --- Simulation Loop ---
    for i in range(N_FRAMES):
        # --- 1. Update State based on RULES ---

        # IF droplet.state == AIRBORNE:
        if droplet['state'] == 'AIRBORNE':
            droplet['velocity_y'] -= gravity
            droplet['y'] += droplet['velocity_y']
            
            if droplet['y'] <= surface_y and droplet['velocity_y'] < 0:
                droplet['y'] = surface_y
                droplet['velocity_y'] = 0
                droplet['state'] = 'COMPRESSING'

        # IF droplet.state == COMPRESSING:
        elif droplet['state'] == 'COMPRESSING':
            droplet['height'] -= deformation_rate
            droplet['width'] += deformation_rate
            
            if droplet['height'] <= droplet['min_height']:
                droplet['state'] = 'REBOUNDING'

        # IF droplet.state == REBOUNDING:
        elif droplet['state'] == 'REBOUNDING':
            droplet['height'] += deformation_rate
            droplet['width'] -= deformation_rate
            
            if droplet['height'] >= droplet['initial_height']:
                droplet['height'] = droplet['initial_height']
                droplet['width'] = droplet['initial_height']
                droplet['state'] = 'AIRBORNE'
                droplet['velocity_y'] = restitution_velocity

        # --- 2. Render Frame ---
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        ax.set_facecolor('#e0f7fa') # Light cyan background
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

        # Render the surface
        ax.axhline(y=surface_y, color='#37474f', linewidth=5, zorder=1)

        # Render the droplet as an ellipse
        ellipse_center_x = 200
        ellipse_center_y = droplet['y'] + droplet['height'] / 2
        
        droplet_ellipse = patches.Ellipse(
            (ellipse_center_x, ellipse_center_y),
            width=droplet['width'],
            height=droplet['height'],
            facecolor='#1976d2', # A nice blue
            edgecolor='white',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(droplet_ellipse)

        # Set title with simulation info
        title = (
            f"Step: {i:03d} | State: {droplet['state']}\n"
            f"y: {droplet['y']:.1f}, vy: {droplet['velocity_y']:.1f}, "
            f"h: {droplet['height']:.1f}, w: {droplet['width']:.1f}"
        )
        ax.set_title(title, fontsize=12, pad=10)
        
        # Save the figure
        filepath = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)

    # --- 3. Create Animated GIF ---
    frame_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png")))
    if not frame_files:
        print("Error: No frames were generated. Skipping GIF creation.")
        return

    frames = [Image.open(f) for f in frame_files]
    gif_path = os.path.join(OUTPUT_DIR, "simulation.gif")
    
    frames[0].save(
        gif_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=GIF_DURATION,
        loop=0  # Loop forever
    )

    # --- 4. Print Summary ---
    print(f"Saved {len(frames)} frames + simulation.gif")

if __name__ == "__main__":
    run_simulation()