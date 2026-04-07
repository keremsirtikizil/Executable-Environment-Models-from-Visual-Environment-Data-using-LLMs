# Provider: fal / Model: google/gemini-2.5-pro
# Episode: full_complex
# Actions: ['RIGHT', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'UP', 'UP']

import copy

def apply_action(grid, agent_pos, action):
    """
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    """
    # Cell type assignments:
    # 0: Empty
    # 1: Agent (blue diamond)
    # 2: Red circle
    # 3: Green triangle
    
    # Action integer meanings:
    # 0: Stay
    # 1: Up
    # 2: Down
    # 3: Left
    # 4: Right

    new_grid = copy.deepcopy(grid)
    new_agent_pos = agent_pos
    height, width = len(new_grid), len(new_grid[0])

    # --- Phase 1: Automatic Grid Update ---
    
    triangle_positions = []
    circle_pos = None
    for r in range(height):
        for c in range(width):
            if new_grid[r][c] == 3: # Green triangle
                triangle_positions.append((r, c))
            elif new_grid[r][c] == 2: # Red circle
                circle_pos = (r, c)
    
    num_triangles = len(triangle_positions)
    if num_triangles > 0 and circle_pos:
        # Remove old circle
        r, c = circle_pos
        new_grid[r][c] = 0 # Empty
        
        # Calculate and place new circle
        new_r, new_c = r + num_triangles, c + num_triangles
        if 0 <= new_r < height and 0 <= new_c < width:
            new_grid[new_r][new_c] = 2 # Red circle
            
        # Remove all triangles
        for tr, tc in triangle_positions:
            new_grid[tr][tc] = 0 # Empty

    # --- Phase 2: Agent Movement ---

    moves = {
        1: (-1, 0),  # Up
        2: (1, 0),   # Down
        3: (0, -1),  # Left
        4: (0, 1),   # Right
    }
    
    if action in moves:
        dr, dc = moves[action]
        r, c = new_agent_pos
        target_r, target_c = r + dr, c + dc

        # Check if target is valid (in-bounds and empty)
        if 0 <= target_r < height and 0 <= target_c < width and new_grid[target_r][target_c] == 0:
            new_grid[r][c] = 0 # Set old position to empty
            new_grid[target_r][target_c] = 1 # Move agent to new position
            new_agent_pos = (target_r, target_c)

    return new_grid, new_agent_pos