"""
Ground-truth evaluator for EchoWorld rule extraction.

The five rules being tested:
  1. Normal movement     — agent moves to empty cell; wall blocks it.
  2. Echo moves opposite — when agent moves, echo simultaneously moves
                           in the OPPOSITE direction.
  3. Blocked echo        — if echo would hit wall/agent, echo stays but agent
                           STILL moves (unlike MagnetWorld).
  4. Void consumes echo  — echo lands on void → both vanish.
  5. Beacon bounce       — agent steps on beacon → bounces one extra
                           step in the same direction (if not blocked by wall/echo).
"""

import copy

_WALL  = 1
_ECHO  = 2
_VOID  = 3

W, E, O, V, B, A = 1, 0, 2, 3, 4, 5  # Wall, Empty, echO, Void, Beacon, Agent


def build_echo_test_cases() -> list:
    """Return the canonical test suite for EchoWorld."""
    return [
        # ── Rule 1: normal movement ─────────────────────────────────
        {
            "name": "move_right_empty",
            "description": "Agent moves right; echo against left wall stays",
            "grid": [
                [W, W, W, W, W, W, W],
                [W, O, E, E, A, E, W],
                [W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,5). Echo at (1,1) opp=LEFT → (1,0)=W → stays.
            "agent_pos": (1, 4), "action": 3,
            "expected_agent_pos": (1, 5),
            "expected_echo_pos": (1, 1),
        },
        {
            "name": "move_left_empty",
            "description": "Agent moves left; echo against right wall stays",
            "grid": [
                [W, W, W, W, W, W, W],
                [W, E, A, E, E, O, W],
                [W, W, W, W, W, W, W],
            ],
            # Agent LEFT → (1,1). Echo at (1,5) opp=RIGHT → (1,6)=W → stays.
            "agent_pos": (1, 2), "action": 2,
            "expected_agent_pos": (1, 1),
            "expected_echo_pos": (1, 5),
        },
        {
            "name": "move_down_empty",
            "description": "Agent moves down; echo against top wall stays",
            "grid": [
                [W, W, W, W, W],
                [W, E, O, E, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            # Agent DOWN → (4,2). Echo at (1,2) opp=UP → (0,2)=W → stays.
            "agent_pos": (3, 2), "action": 1,
            "expected_agent_pos": (4, 2),
            "expected_echo_pos": (1, 2),
        },
        {
            "name": "move_up_blocked",
            "description": "Agent tries to move up into wall — nothing happens",
            "grid": [
                [W, W, W, W, W],
                [W, E, A, E, W],
                [W, E, E, E, W],
                [W, E, E, E, W],
                [W, E, O, E, W],
                [W, W, W, W, W],
            ],
            # Agent UP → (0,2)=W → blocked. Nothing moves.
            "agent_pos": (1, 2), "action": 0,
            "expected_agent_pos": (1, 2),
            "expected_echo_pos": (4, 2),
        },
        {
            "name": "wall_blocks_agent_right",
            "description": "Agent tries to move right into wall",
            "grid": [
                [W, W, W, W, W],
                [W, O, E, A, W],
                [W, W, W, W, W],
            ],
            "agent_pos": (1, 3), "action": 3,
            "expected_agent_pos": (1, 3),
            "expected_echo_pos": (1, 1),
        },

        # ── Rule 2: echo moves opposite ─────────────────────────────
        {
            "name": "echo_moves_opposite_right",
            "description": "Agent RIGHT → echo LEFT into open space",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, E, O, E, A, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,6). Echo at (1,3) opp=LEFT → (1,2).
            "agent_pos": (1, 5), "action": 3,
            "expected_agent_pos": (1, 6),
            "expected_echo_pos": (1, 2),
        },
        {
            "name": "echo_moves_opposite_down",
            "description": "Agent DOWN → echo UP into open space",
            "grid": [
                [W, W, W, W, W],
                [W, E, E, E, W],
                [W, E, O, E, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            # Agent DOWN → (5,2). Echo at (2,2) opp=UP → (1,2).
            "agent_pos": (4, 2), "action": 1,
            "expected_agent_pos": (5, 2),
            "expected_echo_pos": (1, 2),
        },
        {
            "name": "echo_moves_opposite_left",
            "description": "Agent LEFT → echo RIGHT into open space",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, A, E, O, E, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            # Agent LEFT → (1,1). Echo at (1,4) opp=RIGHT → (1,5).
            "agent_pos": (1, 2), "action": 2,
            "expected_agent_pos": (1, 1),
            "expected_echo_pos": (1, 5),
        },
        {
            "name": "echo_moves_opposite_up",
            "description": "Agent UP → echo DOWN into open space",
            "grid": [
                [W, W, W, W, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, E, O, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            # Agent UP → (1,2). Echo at (3,2) opp=DOWN → (4,2).
            "agent_pos": (2, 2), "action": 0,
            "expected_agent_pos": (1, 2),
            "expected_echo_pos": (4, 2),
        },

        # ── Rule 3: blocked echo, agent still moves ─────────────────
        {
            "name": "echo_blocked_by_wall_agent_moves",
            "description": "Agent DOWN; echo would go UP into wall — echo stays, agent moves",
            "grid": [
                [W, W, W, W, W],
                [W, E, O, E, W],
                [W, E, E, E, W],
                [W, E, A, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            # Agent DOWN → (4,2). Echo at (1,2) opp=UP → (0,2)=W → stays.
            "agent_pos": (3, 2), "action": 1,
            "expected_agent_pos": (4, 2),
            "expected_echo_pos": (1, 2),
        },
        {
            "name": "echo_blocked_right_wall",
            "description": "Agent LEFT; echo would go RIGHT into wall — stays",
            "grid": [
                [W, W, W, W, W, W],
                [W, E, A, E, O, W],
                [W, W, W, W, W, W],
            ],
            # Agent LEFT → (1,1). Echo at (1,4) opp=RIGHT → (1,5)=W → stays.
            "agent_pos": (1, 2), "action": 2,
            "expected_agent_pos": (1, 1),
            "expected_echo_pos": (1, 4),
        },
        {
            "name": "echo_blocked_by_agent",
            "description": "Echo would move onto agent's new position — blocked",
            "grid": [
                [W, W, W, W, W, W],
                [W, E, E, A, O, W],
                [W, W, W, W, W, W],
            ],
            # Agent LEFT → (1,2). Echo at (1,4) opp=RIGHT → (1,5)=W → stays.
            # Hmm that's wall again. Let me use a different layout:
            # Agent moves RIGHT, echo is 2 cells right of agent.
            # Agent at (1,2) RIGHT → (1,3). Echo at (1,4) opp=LEFT → (1,3)=AGENT → blocked.
            "grid": [
                [W, W, W, W, W, W],
                [W, E, A, E, O, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 2), "action": 3,
            "expected_agent_pos": (1, 3),
            "expected_echo_pos": (1, 4),
        },

        # ── Rule 4: void consumes echo ──────────────────────────────
        {
            "name": "echo_into_void",
            "description": "Agent RIGHT → echo LEFT into void — both echo and void vanish",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, V, O, E, A, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,6). Echo at (1,3) opp=LEFT → (1,2)=VOID → consumed.
            "agent_pos": (1, 5), "action": 3,
            "expected_agent_pos": (1, 6),
            "expected_echo_pos": None,
        },
        {
            "name": "echo_not_into_void",
            "description": "Void exists but not in echo's path — nothing consumed",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, E, O, E, A, E, W],
                [W, E, V, E, E, E, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,6). Echo at (1,3) opp=LEFT → (1,2)=E. Void on row 2.
            "agent_pos": (1, 5), "action": 3,
            "expected_agent_pos": (1, 6),
            "expected_echo_pos": (1, 2),
        },
        {
            "name": "agent_walks_over_void",
            "description": "Agent walks onto void — no effect (void only affects echo)",
            "grid": [
                [W, W, W, W, W, W, W],
                [W, O, E, E, A, V, W],
                [W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,5)=VOID. Agent stands there.
            # Echo at (1,1) opp=LEFT → (1,0)=W → stays.
            "agent_pos": (1, 4), "action": 3,
            "expected_agent_pos": (1, 5),
            "expected_echo_pos": (1, 1),
        },
        {
            "name": "echo_consumed_no_more_echo",
            "description": "After echo consumed, further moves have no echo to move",
            "grid": [
                [W, W, W, W, W, W],
                [W, V, O, E, A, W],
                [W, W, W, W, W, W],
            ],
            # Agent LEFT → (1,3). Echo at (1,2) opp=RIGHT → (1,3)=AGENT → blocked.
            # Wait: echo blocked by agent. Not consumed. Let me redesign.
            # Agent RIGHT into wall → blocked. Nothing happens.
            # Need echo to land on void. Agent LEFT, echo RIGHT into void?
            # Echo at (1,2), void at (1,1). Agent LEFT → opp=RIGHT. echo → (1,3). Not void.
            # Agent RIGHT, echo LEFT. Echo at (1,2) → (1,1)=VOID. Consumed!
            "grid": [
                [W, W, W, W, W, W],
                [W, V, O, E, A, W],
                [W, W, W, W, W, W],
            ],
            "agent_pos": (1, 4), "action": 3,
            # Agent RIGHT → (1,5)=W → blocked! Nothing happens.
            # Need different layout.
            "grid": [
                [W, W, W, W, W, W, W],
                [W, V, O, E, A, E, W],
                [W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,5). Echo at (1,2) opp=LEFT → (1,1)=VOID → consumed.
            "agent_pos": (1, 4), "action": 3,
            "expected_agent_pos": (1, 5),
            "expected_echo_pos": None,
        },

        # ── Rule 5: beacon bounce ────────────────────────────────────
        {
            "name": "beacon_bounce_right",
            "description": "Agent steps on beacon, bounces one extra step right",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, O, E, E, A, B, E, W],
                [W, W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,5)=BEACON. Echo at (1,1) opp=LEFT → (1,0)=W → stays.
            # Bounce: (1,6)=E, not WALL/ECHO → agent bounces to (1,6).
            "agent_pos": (1, 4), "action": 3,
            "expected_agent_pos": (1, 6),
            "expected_echo_pos": (1, 1),
        },
        {
            "name": "beacon_bounce_blocked_by_wall",
            "description": "Beacon bounce blocked by wall — agent stays on beacon cell",
            "grid": [
                [W, W, W, W, W, W],
                [W, O, E, A, B, W],
                [W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,4)=BEACON. Echo (1,1) opp=LEFT → (1,0)=W → stays.
            # Bounce: (1,5)=W → blocked. Agent stays at (1,4).
            "agent_pos": (1, 3), "action": 3,
            "expected_agent_pos": (1, 4),
            "expected_echo_pos": (1, 1),
        },
        {
            "name": "beacon_bounce_down",
            "description": "Agent steps on beacon moving down, bounces extra step",
            "grid": [
                [W, W, W, W, W],
                [W, E, O, E, W],
                [W, E, A, E, W],
                [W, E, B, E, W],
                [W, E, E, E, W],
                [W, W, W, W, W],
            ],
            # Agent DOWN → (3,2)=BEACON. Echo (1,2) opp=UP → (0,2)=W → stays.
            # Bounce: (4,2)=E → agent bounces to (4,2).
            "agent_pos": (2, 2), "action": 1,
            "expected_agent_pos": (4, 2),
            "expected_echo_pos": (1, 2),
        },
        {
            "name": "beacon_bounce_blocked_by_echo",
            "description": "Beacon bounce blocked by echo — agent stays on beacon",
            "grid": [
                [W, W, W, W, W, W, W, W],
                [W, E, E, A, B, E, O, W],
                [W, W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,4)=BEACON. Echo at (1,6) opp=LEFT → (1,5).
            # Bounce: (1,5) now has ECHO → blocked. Agent stays at (1,4).
            "agent_pos": (1, 3), "action": 3,
            "expected_agent_pos": (1, 4),
            "expected_echo_pos": (1, 5),
        },
        {
            "name": "no_beacon_normal_move",
            "description": "Beacon exists but agent moves onto empty — no bounce",
            "grid": [
                [W, W, W, W, W, W],
                [W, O, E, A, E, W],
                [W, E, E, E, B, W],
                [W, W, W, W, W, W],
            ],
            # Agent RIGHT → (1,4)=E. Echo (1,1) opp=LEFT → (1,0)=W → stays.
            "agent_pos": (1, 3), "action": 3,
            "expected_agent_pos": (1, 4),
            "expected_echo_pos": (1, 1),
        },

        # ── Large grid ──────────────────────────────────────────────
        {
            "name": "large_grid_echo_opposite",
            "description": "10x10 grid — echo moves opposite across open space",
            "grid": [
                [W, W, W, W, W, W, W, W, W, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, O, E, E, E, A, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, E, E, E, E, E, E, E, E, W],
                [W, W, W, W, W, W, W, W, W, W],
            ],
            # Agent RIGHT → (4,7). Echo (4,2) opp=LEFT → (4,1).
            "agent_pos": (4, 6), "action": 3,
            "expected_agent_pos": (4, 7),
            "expected_echo_pos": (4, 1),
        },
    ]


# ---------------------------------------------------------------------------

def evaluate_echo_function(code: str, test_cases: list) -> dict:
    """
    Execute the VLM-extracted apply_action function and run it against test_cases.
    Same interface as MagnetWorld evaluator but checks echo_pos instead of metal_pos.
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

        new_echo_pos = _find_cell(new_grid, _ECHO)

        agent_ok = (new_agent_pos == tc["expected_agent_pos"])
        echo_ok  = (new_echo_pos == tc["expected_echo_pos"])
        passed   = agent_ok and echo_ok
        if passed:
            n_passed += 1

        results.append({
            "name":        tc["name"],
            "description": tc.get("description", ""),
            "passed":      passed,
            "agent_ok":    agent_ok,
            "metal_ok":    echo_ok,       # keep key name for visualizer compat
            "got_agent":   new_agent_pos,
            "got_metal":   new_echo_pos,  # keep key name for visualizer compat
            "exp_agent":   tc["expected_agent_pos"],
            "exp_metal":   tc["expected_echo_pos"],
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


def print_echo_evaluation_report(result: dict, provider: str = ""):
    header = f"EchoWorld Evaluation — {provider}" if provider else "EchoWorld Evaluation"
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
                    print(f"         echo     : got {r['got_metal']}  "
                          f"expected {r['exp_metal']}")

    print("=" * 60 + "\n")
