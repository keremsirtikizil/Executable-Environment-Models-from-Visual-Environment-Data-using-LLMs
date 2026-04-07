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
    """
    # Cell type mapping:
    # 0: Floor (light beige)
    # 1: Wall (dark grey)
    # 2: Red circle
    # The agent (blue diamond) is not represented in the grid itself.

    # Action mapping:
    # 0: Up
    # 1: Down
    # 2: Left
    # 3: Right

    # The grid state is static in this problem, but we copy for good practice.
    new_grid = [row[:] for row in grid]
    
    r, c = agent_pos
    
    # Calculate the target position based on the action
    if action == 0:  # Up
        next_pos = (r - 1, c)
    elif action == 1:  # Down
        next_pos = (r + 1, c)
    elif action == 2:  # Left
        next_pos = (r, c - 1)
    elif action == 3:  # Right
        next_pos = (r, c + 1)
    else:
        # Invalid action, no change
        return new_grid, agent_pos

    next_r, next_c = next_pos

    # Check if the target cell is valid for movement (i.e., empty floor)
    # The agent cannot move into walls (1) or red circles (2).
    if new_grid[next_r][next_c] == 0:
        return new_grid, next_pos
    else:
        # If the move is invalid, the agent stays in the same position
        return new_grid, agent_pos