# Provider: fal / Model: google/gemini-2.5-pro
# Episode: full_echo
# Actions: ['RIGHT', 'DOWN', 'LEFT', 'LEFT', 'LEFT']

def apply_action(grid, agent_pos, action):
    """
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    """
    # Cell type assignments
    EMPTY = 0
    WALL = 1  # Assuming the dark border is a wall
    AGENT = 2
    DIAMOND = 3
    CIRCLE = 4
    STAR = 5

    # Action assignments
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    WAIT = 4

    # Deep copy the grid to avoid modifying the original
    new_grid = [row[:] for row in grid]
    new_agent_pos = agent_pos

    rows, cols = len(new_grid), len(new_grid[0])
    agent_r, agent_c = agent_pos

    # Find positions of other dynamic objects
    diamond_pos, circle_pos = None, None
    for r in range(rows):
        for c in range(cols):
            if new_grid[r][c] == DIAMOND:
                diamond_pos = (r, c)
            elif new_grid[r][c] == CIRCLE:
                circle_pos = (r, c)

    action_deltas = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    if action in action_deltas:
        # Agent directional move
        dr, dc = action_deltas[action]
        next_agent_r, next_agent_c = agent_r + dr, agent_c + dc

        # Check for valid move (within bounds and not into a wall)
        if 0 <= next_agent_r < rows and 0 <= next_agent_c < cols and new_grid[next_agent_r][next_agent_c] != WALL:
            # Move agent
            new_grid[agent_r][agent_c] = EMPTY
            new_grid[next_agent_r][next_agent_c] = AGENT
            new_agent_pos = (next_agent_r, next_agent_c)

            # Diamond moves with the agent
            if diamond_pos:
                dia_r, dia_c = diamond_pos
                new_grid[dia_r][dia_c] = EMPTY
                new_grid[dia_r + dr][dia_c + dc] = DIAMOND

    elif action == WAIT:
        # Agent waits, other objects move
        
        # 1. Green Diamond moves towards the cell right of the agent
        if diamond_pos:
            dia_r, dia_c = diamond_pos
            target_r, target_c = agent_r, agent_c + 1
            
            if (dia_r, dia_c) != (target_r, target_c):
                dr, dc = 0, 0
                # Prioritize horizontal movement
                if dia_c < target_c: dc = 1
                elif dia_c > target_c: dc = -1
                elif dia_r < target_r: dr = 1
                elif dia_r > target_r: dr = -1
                
                if dr != 0 or dc != 0:
                    new_grid[dia_r][dia_c] = EMPTY
                    new_grid[dia_r + dr][dia_c + dc] = DIAMOND

        # 2. Red Circle moves on its patrol path
        if circle_pos:
            circ_r, circ_c = circle_pos
            next_circ_r, next_circ_c = circ_r, circ_c

            # Patrol path is a 3x3 rectangle from (2,5) to (4,7)
            # Top edge: move right
            if circ_r == 2 and circ_c < 7: next_circ_c += 1
            # Right edge: move down
            elif circ_c == 7 and circ_r < 4: next_circ_r += 1
            # Bottom edge: move left
            elif circ_r == 4 and circ_c > 5: next_circ_c -= 1
            # Left edge: move up
            elif circ_c == 5 and circ_r > 2: next_circ_r -= 1

            # The destination cell becomes empty, even if it was a STAR
            new_grid[next_circ_r][next_circ_c] = EMPTY
            
            # Move circle
            new_grid[circ_r][circ_c] = EMPTY
            new_grid[next_circ_r][next_circ_c] = CIRCLE

    return new_grid, new_agent_pos