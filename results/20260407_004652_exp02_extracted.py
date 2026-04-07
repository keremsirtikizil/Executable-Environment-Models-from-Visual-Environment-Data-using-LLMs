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
    - 0: EMPTY
    - 1: AGENT (blue diamond)
    - 2: CIRCLE (red, initial static state)
    - 3: TRIANGLE (green)
    
    To handle the circle's movement, we infer hidden states that are visually
    identical to the static circle but have different behaviors.
    - 4: CIRCLE_DR (red, moving down-right)
    - 5: CIRCLE_UL (red, moving up-left)
    """
    new_grid = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0])

    # Define cell types for clarity
    EMPTY = 0
    AGENT = 1
    CIRCLE_STATIC = 2
    TRIANGLE = 3
    CIRCLE_DR = 4
    CIRCLE_UL = 5

    # --- Phase 1: Check for activation of a static circle ---
    # This event takes precedence over regular movement.
    static_circle_pos = None
    for r in range(height):
        for c in range(width):
            if grid[r][c] == CIRCLE_STATIC:
                static_circle_pos = (r, c)
                break
        if static_circle_pos:
            break

    if static_circle_pos:
        r, c = static_circle_pos
        triangles_to_remove = []
        
        # Scan four cardinal directions for triangles
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            cr, cc = r + dr, c + dc
            while 0 <= cr < height and 0 <= cc < width:
                if grid[cr][cc] == TRIANGLE:
                    triangles_to_remove.append((cr, cc))
                elif grid[cr][cc] != EMPTY:
                    break # Blocked by another object
                cr, cc = cr + dr, cc + dc
        
        n = len(triangles_to_remove)
        if n > 0:
            for tr, tc in triangles_to_remove:
                new_grid[tr][tc] = EMPTY
            
            new_r, new_c = r + n, c + n
            new_grid[r][c] = EMPTY
            if 0 <= new_r < height and 0 <= new_c < width:
                new_grid[new_r][new_c] = CIRCLE_DR
            
            return (new_grid, agent_pos)

    # --- Phase 2: If no activation, process regular movement ---
    moves = []
    for r in range(height):
        for c in range(width):
            cell = grid[r][c]
            
            if cell == CIRCLE_DR:
                nr, nc = r + 1, c + 1
                # Bounce if next position is invalid (out of bounds or occupied)
                if not (0 <= nr < height and 0 <= nc < width and grid[nr][nc] == EMPTY):
                    moves.append(((r, c), (r - 1, c - 1), CIRCLE_UL))
                else:
                    moves.append(((r, c), (nr, nc), CIRCLE_DR))

            elif cell == CIRCLE_UL:
                nr, nc = r - 1, c - 1
                # Bounce if next position is invalid
                if not (0 <= nr < height and 0 <= nc < width and grid[nr][nc] == EMPTY):
                    moves.append(((r, c), (r + 1, c + 1), CIRCLE_DR))
                else:
                    moves.append(((r, c), (nr, nc), CIRCLE_UL))
    
    # Apply all collected moves to the new grid
    for (r, c), (nr, nc), new_type in moves:
        new_grid[r][c] = EMPTY
        if 0 <= nr < height and 0 <= nc < width:
            new_grid[nr][nc] = new_type
            
    return (new_grid, agent_pos)