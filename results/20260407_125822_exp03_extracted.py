# Provider: fal / Model: google/gemini-2.5-pro
# Episode: full_echo
# Actions: ['LEFT', 'UP', 'UP', 'DOWN', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'LEFT', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'UP', 'UP', 'UP', 'UP', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'RIGHT', 'DOWN', 'LEFT', 'LEFT', 'UP']

def apply_action(grid, agent_pos, action):
    """
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    """
    import copy

    # Cell type assignments
    EMPTY = 0
    AGENT = 1
    DIAMOND = 3
    TARGET = 4
    STAR = 5

    # Action mapping: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
    moves = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    new_grid = copy.deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    dr, dc = moves.get(action, (0, 0))
    if dr == 0 and dc == 0 and action not in moves: # No-op for invalid action
        return new_grid, agent_pos

    # 1. Calculate and validate agent's move
    r, c = agent_pos
    next_r, next_c = r + dr, c + dc

    if not (0 <= next_r < height and 0 <= next_c < width):
        return grid, agent_pos  # Agent move is out of bounds

    destination_cell = grid[next_r][next_c]
    if destination_cell in [DIAMOND, TARGET]:
        return grid, agent_pos  # Agent is blocked

    # If we reach here, the agent's move is valid.
    new_agent_pos = (next_r, next_c)

    # 2. Find and process the diamond's move
    diamond_pos = None
    for i in range(height):
        for j in range(width):
            if grid[i][j] == DIAMOND:
                diamond_pos = (i, j)
                break
        if diamond_pos:
            break

    if diamond_pos:
        dr_d, dc_d = diamond_pos
        next_r_d, next_c_d = dr_d + dr, dc_d + dc

        if 0 <= next_r_d < height and 0 <= next_c_d < width:
            diamond_dest_cell = grid[next_r_d][next_c_d]
            if diamond_dest_cell == TARGET:
                new_grid[dr_d][dc_d] = EMPTY
                new_grid[next_r_d][next_c_d] = EMPTY
            elif diamond_dest_cell == EMPTY:
                new_grid[dr_d][dc_d] = EMPTY
                new_grid[next_r_d][next_c_d] = DIAMOND

    # 3. Execute the agent's move
    new_grid[r][c] = EMPTY
    new_grid[next_r][next_c] = AGENT  # This overwrites stars, effectively collecting them

    return new_grid, new_agent_pos