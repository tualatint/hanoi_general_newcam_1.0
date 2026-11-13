"""Tower-of-Hanoi state model, BFS solver, and initial-state sampler.

Conventions
-----------
* There are 3 disks indexed by SIZE: 0 = large (bottom), 1 = medium, 2 = small.
* A state is ``(peg_of_large, peg_of_medium, peg_of_small)`` with pegs
  ``0, 1, 2``.  A state is legal iff every peg holds its disks in size order;
  because the size order on a peg is forced (larger below smaller), *every*
  assignment of disks to pegs is physically legal -- the legal set is all
  ``3 ** 3 = 27`` assignments.
* ``GOAL = (2, 2, 2)``: the classic "move the whole tower to peg C" task.
* ``next_move`` returns the deterministic optimal move from a state, using the
  BFS distance-to-goal table.  Ties are broken by moving the SMALLEST disk
  first, which reproduces the textbook solution from the canonical
  all-on-A state (small A->C, medium A->B, ...).
"""
from __future__ import annotations

from collections import deque
from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

N_DISKS = 3
N_PEGS = 3
DISK_NAMES = ("disk_l", "disk_m", "disk_s")  # index 0,1,2
DISK_LABELS = ("L", "M", "S")
PEG_NAMES = ("A", "B", "C")
GOAL_PEG = 2
GOAL_STATE = (GOAL_PEG,) * N_DISKS

Move = Tuple[int, int, int]  # (disk, src_peg, dst_peg)
State = Tuple[int, int, int]


def top_disk(state: Sequence[int], peg: int) -> Optional[int]:
    """Smallest (topmost) disk on a peg, or None.

    Disk index 0 is the LARGEST disk, so the top disk is the MAXIMUM index on
    the peg."""
    on = [d for d in range(N_DISKS) if state[d] == peg]
    return max(on) if on else None


def legal_moves(state: State) -> List[Move]:
    """All legal single-disk transfers: take the top disk and put it on an
    empty peg or on a larger disk."""
    out: List[Move] = []
    for src in range(N_PEGS):
        d = top_disk(state, src)
        if d is None:
            continue
        for dst in range(N_PEGS):
            if dst == src:
                continue
            t = top_disk(state, dst)
            if t is None or d > t:  # moving disk is smaller than dst top
                out.append((d, src, dst))
    return out


def apply_move(state: State, move: Move) -> State:
    d, _src, dst = move
    s = list(state)
    s[d] = dst
    return tuple(s)


def _bfs(goal: State = GOAL_STATE):
    dist = {goal: 0}
    dq = deque([goal])
    while dq:
        s = dq.popleft()
        # The move graph is undirected (any transfer can be undone), so
        # expanding the goal in the same way gives distances to the goal.
        for mv in legal_moves(s):
            ns = apply_move(s, mv)
            if ns not in dist:
                dist[ns] = dist[s] + 1
                dq.append(ns)
    return dist


_DIST = _bfs(GOAL_STATE)
_DIST_CACHE = {GOAL_STATE: _DIST}


def _table_for(goal: State) -> dict:
    goal = tuple(int(v) for v in goal)
    if goal not in _DIST_CACHE:
        _DIST_CACHE[goal] = _bfs(goal)
    return _DIST_CACHE[goal]


def distance(state: State, goal: State = GOAL_STATE) -> int:
    """Optimal number of moves from ``state`` to ``goal``.

    Raises KeyError for an unreachable state (cannot happen for the 27 legal
    assignments)."""
    return int(_table_for(goal)[tuple(int(v) for v in state)])


def distance_table(goal: State = GOAL_STATE) -> dict:
    return dict(_table_for(goal))


def next_move(state: State, goal: State = GOAL_STATE) -> Optional[Move]:
    """Deterministic optimal first move from ``state`` (None at the goal)."""
    state = tuple(int(v) for v in state)
    table = _table_for(goal)
    d0 = table[state]
    if d0 == 0:
        return None
    cands = []
    for mv in legal_moves(state):
        ns = apply_move(state, mv)
        if table[ns] == d0 - 1:
            # smallest disk first (highest disk index), then lowest dst peg
            cands.append((mv[0], mv[2], mv))
    if not cands:
        raise RuntimeError(f"no optimal move from {state}")
    cands.sort(reverse=True)
    return cands[0][2]


def plan(state: State, goal: State = GOAL_STATE) -> List[Move]:
    """Full optimal move list from ``state`` to ``goal``."""
    out: List[Move] = []
    s = tuple(int(v) for v in state)
    while True:
        mv = next_move(s, goal=goal)
        if mv is None:
            return out
        out.append(mv)
        s = apply_move(s, mv)
        if len(out) > 100:
            raise RuntimeError("plan did not terminate")


def all_legal_states() -> List[State]:
    # Every assignment is legal (the physical stack order is forced).
    return [tuple(s) for s in product(range(N_PEGS), repeat=N_DISKS)]


def long_states(min_dist: int = 5, goal: State = GOAL_STATE) -> List[State]:
    """Initial states whose optimal solution is at least ``min_dist`` moves.

    For 3 disks the counts are d=0:1, 1:2, 2:2, 3:4, 4:2, 5:4, 6:4, 7:8, so
    ``min_dist=5`` gives 16 states (4 five-move, 4 six-move, 8 seven-move).
    """
    if goal != GOAL_STATE:
        table = _bfs(goal)
    else:
        table = _DIST
    return [s for s in all_legal_states() if table[s] >= min_dist]


def sample_state(rng, min_dist: int = 5,
                 goal: State = GOAL_STATE) -> State:
    """Uniform sample from the long states (works with both the legacy
    ``RandomState`` used by the env and a modern ``Generator``)."""
    pool = long_states(min_dist=min_dist, goal=goal)
    idx = int(rng.choice(len(pool)))
    return tuple(pool[idx])


def state_to_str(state: Sequence[int]) -> str:
    """Human-readable like ``L:A M:C S:B``."""
    return " ".join(
        f"{DISK_LABELS[d]}:{PEG_NAMES[int(state[d])]}" for d in range(N_DISKS))


if __name__ == "__main__":
    from collections import Counter

    print("legal states:", len(all_legal_states()))
    print("distance histogram:", dict(sorted(Counter(_DIST.values()).items())))
    for s in long_states(7):
        print(state_to_str(s), "d=", distance(s),
              "plan=", [(DISK_LABELS[d], PEG_NAMES[a], PEG_NAMES[b])
                        for d, a, b in plan(s)])
