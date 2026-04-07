# Provider: fal / Model: google/gemini-2.5-pro
# Episode: full_complex
# Actions: ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'UP', 'UP', 'UP', 'UP', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'DOWN', 'DOWN', 'LEFT', 'DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP', 'UP', 'LEFT', 'LEFT', 'LEFT', 'UP', 'UP']

def apply_action(grid, agent_pos, action):
    """
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    0: Empty (beige)
    3: Black Square
    4: Green Triangle
    5: Blue Diamond
    6: Orange X-Box
    The agent (red circle) is represented by agent_pos.
    The outer wall is a boundary, not a grid value.

    Actions:
    0: Up
    1: Down
    2: Left
    3: Right
    """
    import copy
    new_grid = copy.deepcopy(grid)
    r, c = agent_pos
    
    # Define cell types and actions for clarity
    EMPTY = 0
    TRIANGLE = 4
    DIAMOND = 5
    X_BOX = 6
    
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    # Calculate target position
    dr, dc = 0, 0
    if action == UP:
        dr = -1
    elif action == DOWN:
        dr = 1
    elif action == LEFT:
        dc = -1
    elif action == RIGHT:
        dc = 1
    
    target_r, target_c = r + dr, c + dc

    # Check for valid move (bounds only)
    height = len(new_grid)
    width = len(new_grid[0])
    if not (0 <= target_r < height and 0 <= target_c < width):
        return (new_grid, agent_pos)

    # If move is valid, update agent position
    new_agent_pos = (target_r, target_c)

    # Handle interactions at the new position
    if new_grid[target_r][target_c] == TRIANGLE:
        new_grid[target_r][target_c] = EMPTY

    # Special rule: successful horizontal moves clear diamonds and x-boxes
    if action == LEFT or action == RIGHT:
        for i in range(height):
            for j in range(width):
                if new_grid[i][j] in [DIAMOND, X_BOX]:
                    new_grid[i][j] = EMPTY
    
    return (new_grid, new_agent_pos)