# Provider: fal / Model: google/gemini-2.5-pro
# Episode: full_complex
# Actions: ['RIGHT', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'UP', 'UP']

def apply_action(grid, agent_pos, action):
    """
    grid:      list[list[int]]          deep-copy before modifying
    agent_pos: tuple[int, int]          (row, col)
    action:    int                      (you must infer what each integer means)
    Returns:   (new_grid, new_agent_pos)

    Assign integer values to each visually distinct cell type you observe.
    0: Empty
    1: Agent (blue diamond)
    2: Red circle
    3: Green triangle
    """
    new_grid = [row[:] for row in grid]
    new_agent_pos = agent_pos  # Agent position is static in this environment

    # Action 1 triggers the environment's physics update.
    # Any other action is a no-op.
    if action == 1:
        rows = len(new_grid)
        cols = len(new_grid[0])

        # Iterate through the grid to find a red circle that can jump.
        # Process only the first jump found, then stop.
        for r in range(rows):
            for c in range(cols):
                if new_grid[r][c] == 2:  # Found a red circle
                    # Count consecutive green triangles to the right
                    num_triangles = 0
                    scan_col = c + 1
                    while scan_col < cols and new_grid[r][scan_col] == 3:
                        num_triangles += 1
                        scan_col += 1

                    if num_triangles > 0:
                        # A jump is possible. Calculate destination.
                        dest_r, dest_c = r + 1, c + num_triangles + 1

                        # Check if destination is in bounds and empty
                        if 0 <= dest_r < rows and 0 <= dest_c < cols and new_grid[dest_r][dest_c] == 0:
                            # Perform the jump and removal
                            new_grid[dest_r][dest_c] = 2  # Move circle
                            new_grid[r][c] = 0  # Clear original spot

                            # Remove jumped triangles
                            for i in range(num_triangles):
                                new_grid[r][c + 1 + i] = 0

                            # Only one jump occurs per action
                            return new_grid, new_agent_pos

    # If action is not 1, or no jump was possible, return the current state
    return new_grid, new_agent_pos