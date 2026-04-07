# Provider: fal / Model: google/gemini-2.5-pro
# Episode: attract_3steps
# Actions: ['DOWN', 'DOWN', 'DOWN', 'RIGHT', 'UP', 'RIGHT', 'RIGHT', 'UP', 'LEFT', 'LEFT', 'LEFT', 'RIGHT', 'LEFT']

def apply_action(grid, agent_pos, action):
    """
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    Cell Types:
    0: Floor (beige)
    1: Wall (dark grey)
    2: Red Circle (Mirror mode)
    3: Red Circle (Rotate mode) - inferred state, visually identical to 2

    Actions:
    0: UP
    1: DOWN
    2: LEFT
    3: RIGHT
    4: STAY
    """
    new_grid = [row[:] for row in grid]
    new_agent_pos = agent_pos

    # Constants for cell types
    FLOOR = 0
    WALL = 1
    CIRCLE_MIRROR = 2
    CIRCLE_ROTATE = 3

    # Find the circle's position and type
    circle_pos = None
    circle_type = None
    for r, row in enumerate(new_grid):
        for c, cell in enumerate(row):
            if cell in [CIRCLE_MIRROR, CIRCLE_ROTATE]:
                circle_pos = (r, c)
                circle_type = cell
                break
        if circle_pos:
            break
    
    if circle_pos is None:
        # Should not happen in valid scenarios
        return new_grid, new_agent_pos

    # Action: STAY (toggles circle mode)
    if action == 4:
        if circle_type == CIRCLE_MIRROR:
            new_grid[circle_pos[0]][circle_pos[1]] = CIRCLE_ROTATE
        else:
            new_grid[circle_pos[0]][circle_pos[1]] = CIRCLE_MIRROR
        return new_grid, new_agent_pos

    # Define movement vectors
    moves = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }
    dr, dc = moves[action]

    # 1. Calculate agent's potential move
    next_agent_r, next_agent_c = agent_pos[0] + dr, agent_pos[1] + dc
    
    # 2. Check for agent collision with wall
    if new_grid[next_agent_r][next_agent_c] == WALL:
        return new_grid, new_agent_pos  # No movement for anyone

    # 3. Agent move is valid, update agent position
    new_agent_pos = (next_agent_r, next_agent_c)

    # 4. Determine circle's move based on its mode
    if circle_type == CIRCLE_MIRROR:
        # Mirror mode: circle tries to move in the same direction
        dcr, dcc = dr, dc
    else:  # circle_type == CIRCLE_ROTATE
        # Rotate mode: circle's move is agent's move rotated 90 deg CCW
        dcr, dcc = -dc, dr

    # 5. Calculate circle's potential move
    cr, cc = circle_pos
    next_circle_r, next_circle_c = cr + dcr, cc + dcc

    # 6. Check for circle collision and update grid
    new_grid[cr][cc] = FLOOR
    if new_grid[next_circle_r][next_circle_c] == WALL:
        # Circle move is blocked, it stays in place
        new_grid[cr][cc] = circle_type
    else:
        # Circle move is valid
        new_grid[next_circle_r][next_circle_c] = circle_type

    return new_grid, new_agent_pos