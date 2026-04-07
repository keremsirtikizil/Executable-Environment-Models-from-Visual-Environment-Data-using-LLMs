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
    - 0: Empty cell (beige)
    - 1: Agent (red circle)
    - 2: Blue diamond
    - 3: Green triangle

    Action mapping (8-directional movement):
    - 0: UP (-1, 0)
    - 1: DOWN (1, 0)
    - 2: LEFT (0, -1)
    - 3: RIGHT (0, 1)
    - 4: UP_LEFT (-1, -1)
    - 5: UP_RIGHT (-1, 1)
    - 6: DOWN_LEFT (1, -1)
    - 7: DOWN_RIGHT (1, 1)
    """
    new_grid = copy.deepcopy(grid)
    
    EMPTY = 0
    AGENT = 1
    TRIANGLE = 3

    triangle_count = 0
    for r in range(len(new_grid)):
        for c in range(len(new_grid[0])):
            if new_grid[r][c] == TRIANGLE:
                triangle_count += 1

    if triangle_count == 0:
        return new_grid, agent_pos

    moves = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (-1, -1),
        5: (-1, 1),
        6: (1, -1),
        7: (1, 1)
    }
    
    dr, dc = moves[action]
    distance = triangle_count
    
    r, c = agent_pos
    new_r, new_c = r + distance * dr, c + distance * dc
    
    # Move the agent
    new_grid[r][c] = EMPTY
    new_grid[new_r][new_c] = AGENT
    
    # Remove all triangles
    for r_idx in range(len(new_grid)):
        for c_idx in range(len(new_grid[0])):
            if new_grid[r_idx][c_idx] == TRIANGLE:
                new_grid[r_idx][c_idx] = EMPTY
                
    return new_grid, (new_r, new_c)