"""
Ground-truth evaluator for MagnetWorld rule extraction.

The five rules being tested:
  1. Normal movement     — agent moves to empty cell; wall blocks it.
  2. Magnetic attraction — agent moves toward metal → both shift one step
                           in that direction simultaneously.
  3. Blocked attraction  — if metal's new cell would be a wall, the entire
                           move is cancelled (neither object moves).
  4. Ice slide           — agent steps onto ice (non-attraction) → slides
                           until hitting non-ice cell, wall, or metal.
  5. Hole consumes metal — metal pushed into hole → both vanish.
"""

import copy

# Cell values (mirrored from magnet_env.magnet_world to keep eval self-contained)
_WALL  = 1
_METAL = 2
_ICE   = 3
_HOLE  = 4

W, E, M, A, I, H = 1, 0, 2, 5, 3, 4   # shorthand for grid literals


def build_test_cases() -> list:
    """Return the canonical test suite covering all three rules."""
    return [
        # ── Rule 1: normal movement ─────────────────────────────────────
        {
            "name": "move_right_empty",
            "description": "Agent moves right; metal is directly above (not in path) — basic movement",
            "grid": [
                [W, W, W, W, W],
                [W, M, E, E, W],  # metal above agent, same col → moving right is NOT toward
                [W, A, E, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (2, 1), "action": 3,          # RIGHT
            "expected_agent_pos": (2, 2),
            "expected_metal_pos": (1, 1),               # metal untouched
        },
        {
            "name": "move_down_empty",
            "description": "Agent moves down; metal is to the left — basic movement",
            "grid": [
                [W, W, W, W, W],
                [W, E, E, E, W],
                [W, M, A, E, W],  # metal to the LEFT, moving down is NOT toward
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (2, 2), "action": 1,          # DOWN
            "expected_agent_pos": (3, 2),
            "expected_metal_pos": (2, 1),
        },
        {
            "name": "wall_blocks_agent",
            "description": "Agent cannot move into wall",
            "grid": [
                [W, W, W, W, W],
                [W, M, E, E, W],  # metal above, not in movement direction
                [W, A, W, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (2, 1), "action": 3,          # RIGHT into wall
            "expected_agent_pos": (2, 1),
            "expected_metal_pos": (1, 1),
        },

        # ── Rule 2: magnetic attraction ─────────────────────────────────
        {
            "name": "toward_right_both_move",
            "description": "Agent moves right toward metal — both shift right",
            "grid": [
                [W, W, W, W, W, W, W],
                [W, E, A, E, M, E, W],
                [W, W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,          # RIGHT
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (1, 5),
        },
        {
            "name": "toward_down_both_move",
            "description": "Agent moves down toward metal below — both shift down",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, E, W],
                [W, E, E, E, W],
                [W, E, M, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 1,          # DOWN
            "expected_agent_pos": (2, 2),
            "expected_metal_pos": (4, 2),
        },
        {
            "name": "toward_left_both_move",
            "description": "Agent moves left toward metal — both shift left",
            "grid": [
                [W, W, W, W, W, W, W],
                [W, E, M, E, A, E, W],
                [W, W, W, W, W, W, W],
            ],
            "agent_pos": (1, 4), "action": 2,          # LEFT
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (1, 1),
        },
        {
            "name": "toward_up_both_move",
            "description": "Agent moves up toward metal above — both shift up",
            "grid": [
                [W, W, W, W, W],
                [W, E, E, E, W],
                [W, E, M, E, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (4, 2), "action": 0,          # UP
            "expected_agent_pos": (3, 2),
            "expected_metal_pos": (1, 2),
        },

        # ── Rule 3: blocked attraction ──────────────────────────────────
        {
            "name": "metal_hits_wall_right_cancel",
            "description": "Metal would hit right wall — entire move cancelled",
            "grid": [
                [W, W, W, W, W, W],
                [W, E, A, E, M, W],  # metal at (1,4); wall at (1,5)
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,          # RIGHT
            "expected_agent_pos": (1, 2),
            "expected_metal_pos": (1, 4),
        },
        {
            "name": "metal_hits_wall_down_cancel",
            "description": "Metal would hit bottom wall — entire move cancelled",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, E, W],
                [W, E, M, E, W],
                [W, W, W, W, W],   # metal at (2,2); wall at (3,2)
            ],
            "agent_pos": (1, 2), "action": 1,          # DOWN
            "expected_agent_pos": (1, 2),
            "expected_metal_pos": (2, 2),
        },

        # ── Moving away / orthogonal (metal stays) ──────────────────────
        {
            "name": "move_away_metal_stationary",
            "description": "Agent moves right away from metal on left — metal stays",
            "grid": [
                [W, W, W, W, W, W],
                [W, M, E, A, E, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 3), "action": 3,          # RIGHT (away from metal)
            "expected_agent_pos": (1, 4),
            "expected_metal_pos": (1, 1),
        },
        {
            "name": "orthogonal_metal_stationary",
            "description": "Agent moves down; metal is to the right on the same row — metal stays",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, M, W],
                [W, E, E, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 1,          # DOWN; metal same row to right
            "expected_agent_pos": (2, 2),
            "expected_metal_pos": (1, 3),
        },

        # ── More basic movement (left / up-away) ────────────────────────
        {
            "name": "move_left_empty",
            "description": "Agent moves left; metal is to the right (away) — metal stays",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, M, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 2,          # LEFT, metal to right = away
            "expected_agent_pos": (1, 1),
            "expected_metal_pos": (1, 3),
        },
        {
            "name": "move_up_empty_away",
            "description": "Agent moves up; metal is below (away) — metal stays",
            "grid": [
                [W, W, W, W, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, E, M, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (2, 2), "action": 0,          # UP, metal below = away
            "expected_agent_pos": (1, 2),
            "expected_metal_pos": (3, 2),
        },

        # ── Wall blocks — remaining 3 directions ────────────────────────
        {
            "name": "wall_blocks_left",
            "description": "Agent cannot move left into wall",
            "grid": [
                [W, W, W, W, W],
                [W, A, E, E, W],
                [W, E, M, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 1), "action": 2,          # LEFT into wall at (1,0)
            "expected_agent_pos": (1, 1),
            "expected_metal_pos": (2, 2),
        },
        {
            "name": "wall_blocks_up",
            "description": "Agent cannot move up into wall",
            "grid": [
                [W, W, W, W, W],
                [W, A, E, E, W],
                [W, E, E, E, W],
                [W, E, E, M, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 1), "action": 0,          # UP into wall at (0,1)
            "expected_agent_pos": (1, 1),
            "expected_metal_pos": (3, 3),
        },
        {
            "name": "wall_blocks_down",
            "description": "Agent cannot move down into wall",
            "grid": [
                [W, W, W, W, W],
                [W, M, E, E, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (3, 2), "action": 1,          # DOWN into wall at (4,2)
            "expected_agent_pos": (3, 2),
            "expected_metal_pos": (1, 1),
        },

        # ── Adjacent attraction (agent directly next to metal) ───────────
        {
            "name": "adjacent_toward_right",
            "description": "Agent directly adjacent to metal, moves right — both shift",
            "grid": [
                [W, W, W, W, W, W],
                [W, E, A, M, E, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,          # RIGHT; metal is right next door
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (1, 4),
        },
        {
            "name": "adjacent_toward_down",
            "description": "Agent directly adjacent to metal below, moves down — both shift",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, E, W],
                [W, E, M, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 1,          # DOWN; metal directly below
            "expected_agent_pos": (2, 2),
            "expected_metal_pos": (3, 2),
        },

        # ── Blocked attraction — remaining 2 directions ─────────────────
        {
            "name": "metal_hits_wall_left_cancel",
            "description": "Metal would hit left wall — entire move cancelled",
            "grid": [
                [W, W, W, W, W, W],
                [W, M, E, A, E, W],  # metal at (1,1); wall at (1,0)
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 3), "action": 2,          # LEFT toward metal
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (1, 1),
        },
        {
            "name": "metal_hits_wall_up_cancel",
            "description": "Metal would hit top wall — entire move cancelled",
            "grid": [
                [W, W, W, W, W],
                [W, E, M, E, W],  # metal at (1,2); wall at (0,2)
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (3, 2), "action": 0,          # UP toward metal
            "expected_agent_pos": (3, 2),
            "expected_metal_pos": (1, 2),
        },

        # ── Orthogonal — metal directly below, agent moves right ─────────
        {
            "name": "orthogonal_right_metal_below",
            "description": "Agent moves right; metal is directly below (same col) — metal stays",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, E, W],
                [W, E, M, E, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,          # RIGHT; metal same col below
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (2, 2),
        },
    ]


def build_complex_test_cases() -> list:
    """Extended test suite covering all 5 rules (original 3 + ICE slide + HOLE consume)."""
    base = build_test_cases()
    complex_cases = [
        # ── Rule 4: ICE slide ──────────────────────────────────────────────
        {
            "name": "ice_slide_right",
            "description": "Agent steps onto ice, slides right until hitting empty floor",
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, M, E, E, E, E, E, E, E, W],
                [W, E, A, I, I, I, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "agent_pos": (2, 2), "action": 3,       # RIGHT onto ice; metal at (1,1): mc=1 <= ac=2 → not toward
            "expected_agent_pos": (2, 6),            # slides through 3 ice, stops at empty
            "expected_metal_pos": (1, 1),
        },
        {
            "name": "ice_slide_into_wall",
            "description": "Agent slides on ice and stops just before wall",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, A, I, I, I, I, W],
                [W, M, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,       # RIGHT; metal at (2,1): mc=1 <= ac=2 → not toward
            "expected_agent_pos": (1, 6),            # slides to last ice cell before wall
            "expected_metal_pos": (2, 1),
        },
        {
            "name": "ice_slide_left",
            "description": "Agent slides left on ice, metal is to the right — not toward",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, I, I, I, A, M, W],
                [W, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            "agent_pos": (1, 5), "action": 2,       # LEFT; metal at (1,6): mc=6 >= ac=5 → not toward
            "expected_agent_pos": (1, 1),            # slides all the way to empty
            "expected_metal_pos": (1, 6),
        },
        {
            "name": "ice_slide_down",
            "description": "Agent slides down on vertical ice strip; metal above — not toward",
            "grid": [
                [W, W, W, W, W, W],
                [W, M, A, E, E, W],
                [W, E, I, E, E, W],
                [W, E, I, E, E, W],
                [W, E, I, E, E, W],
                [W, E, E, E, E, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 1,       # DOWN; metal at (1,1): mr=1 <= ar=1 → not toward
            "expected_agent_pos": (5, 2),            # slides through 3 ice, stops at empty
            "expected_metal_pos": (1, 1),
        },
        {
            "name": "no_ice_normal_move",
            "description": "Agent moves onto empty cell next to ice — no sliding",
            "grid": [
                [W, W, W, W, W, W],
                [W, M, A, E, I, W],
                [W, E, E, E, E, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,       # RIGHT onto empty; metal at (1,1): mc=1 <= ac=2 → not toward
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (1, 1),
        },

        # ── Rule 5: HOLE consumes metal ────────────────────────────────────
        {
            "name": "metal_pushed_into_hole",
            "description": "Agent moves toward metal, metal lands on hole — both vanish",
            "grid": [
                [W, W, W, W, W, W, W],
                [W, E, A, E, M, H, W],
                [W, W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,       # RIGHT toward metal
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": None,              # metal consumed by hole
            "consumed_holes": [(1, 5)],              # hole at (1,5) also vanishes
        },
        {
            "name": "metal_pushed_not_into_hole",
            "description": "Agent pushes metal but hole is not in metal's path — metal moves normally",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, A, E, M, E, E, W],
                [W, E, E, E, E, H, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,       # RIGHT toward metal
            "expected_agent_pos": (1, 3),
            "expected_metal_pos": (1, 5),            # normal attraction, hole is on different row
        },
        {
            "name": "agent_walks_over_hole",
            "description": "Agent walks onto hole cell — hole does NOT affect agent",
            "grid": [
                [W, W, W, W, W, W],
                [W, M, A, H, E, W],
                [W, E, E, E, E, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,       # RIGHT onto hole; metal at (1,1): mc=1 <= ac=2 → not toward
            "expected_agent_pos": (1, 3),            # agent just walks over it
            "expected_metal_pos": (1, 1),
        },

        # ── Combined: larger grid with multiple features ────────────────────
        {
            "name": "large_grid_attract_through_corridor",
            "description": "10x10 grid with internal walls — attraction works through open corridor",
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, W, W, W, E, W, W, E, W],
                [W, E, W, E, E, E, E, W, E, W],
                [W, E, W, E, A, E, E, W, E, W],
                [W, E, W, E, E, E, E, W, E, W],
                [W, E, W, E, E, M, E, W, E, W],
                [W, E, W, W, W, W, W, W, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "agent_pos": (4, 4), "action": 1,       # DOWN toward metal
            "expected_agent_pos": (4, 4),            # metal would hit wall at (7,5) — cancel!
            "expected_metal_pos": (6, 5),
        },
        {
            "name": "large_grid_ice_slide",
            "description": "10x10 grid — agent slides on ice, metal is behind (not toward)",
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, I, I, I, I, E, E, W],
                [W, M, A, I, E, E, I, E, E, W],
                [W, E, E, I, I, I, I, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            "agent_pos": (4, 2), "action": 3,       # RIGHT; metal at (4,1): mc=1 <= ac=2 → not toward
            "expected_agent_pos": (4, 4),            # slides on ice, stops at empty (4,4)
            "expected_metal_pos": (4, 1),
        },
    ]
    return base + complex_cases


# ---------------------------------------------------------------------------

def evaluate_extracted_function(code: str, test_cases: list) -> dict:
    """
    Execute the VLM-extracted apply_action function and run it against test_cases.

    Returns dict with keys:
      accuracy    float  fraction of tests passed
      n_passed    int
      n_total     int
      results     list[dict]  per-test details
      error       str | None  top-level compile/exec error if any
    """
    namespace = {}
    try:
        exec(compile(code, "<extracted>", "exec"), namespace)
    except Exception as exc:
        return {
            "accuracy": 0.0,
            "n_passed": 0,
            "n_total": len(test_cases),
            "error": f"Compile/exec error: {exc}",
            "results": [],
        }

    apply_action = namespace.get("apply_action")
    if apply_action is None:
        return {
            "accuracy": 0.0,
            "n_passed": 0,
            "n_total": len(test_cases),
            "error": "apply_action function not found in extracted code",
            "results": [],
        }

    results = []
    n_passed = 0

    for tc in test_cases:
        grid_copy = copy.deepcopy(tc["grid"])
        try:
            new_grid, new_agent_pos = apply_action(
                grid_copy, tc["agent_pos"], tc["action"]
            )
        except Exception as exc:
            results.append({
                "name": tc["name"],
                "description": tc.get("description", ""),
                "passed": False,
                "error": str(exc),
            })
            continue

        new_metal_pos = _find_cell(new_grid, _METAL)

        agent_ok = (new_agent_pos == tc["expected_agent_pos"])
        metal_ok = (new_metal_pos == tc["expected_metal_pos"])
        passed   = agent_ok and metal_ok
        if passed:
            n_passed += 1

        results.append({
            "name":        tc["name"],
            "description": tc.get("description", ""),
            "passed":      passed,
            "agent_ok":    agent_ok,
            "metal_ok":    metal_ok,
            "got_agent":   new_agent_pos,
            "got_metal":   new_metal_pos,
            "exp_agent":   tc["expected_agent_pos"],
            "exp_metal":   tc["expected_metal_pos"],
            "error":       None,
        })

    return {
        "accuracy": n_passed / len(test_cases) if test_cases else 0.0,
        "n_passed": n_passed,
        "n_total":  len(test_cases),
        "error":    None,
        "results":  results,
    }


def _find_cell(grid, value):
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == value:
                return (r, c)
    return None


# ---------------------------------------------------------------------------

def print_evaluation_report(result: dict, provider: str = ""):
    header = f"Evaluation Report — {provider}" if provider else "Evaluation Report"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    if result.get("error"):
        print(f"  FATAL ERROR: {result['error']}")
        return

    print(f"  Passed: {result['n_passed']} / {result['n_total']}"
          f"  ({result['accuracy'] * 100:.1f}%)\n")

    for r in result["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")
        if not r["passed"]:
            if r.get("error"):
                print(f"         error    : {r['error']}")
            else:
                if not r.get("agent_ok"):
                    print(f"         agent    : got {r['got_agent']}  "
                          f"expected {r['exp_agent']}")
                if not r.get("metal_ok"):
                    print(f"         metal    : got {r['got_metal']}  "
                          f"expected {r['exp_metal']}")

    print("=" * 60 + "\n")
