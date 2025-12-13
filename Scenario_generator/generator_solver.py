from __future__ import annotations
from typing import List, Tuple, Optional
import time
from z3 import (
    Solver, Bool, Int, And, Or, Not, If, Implies, Sum, is_true
)

Coord = Tuple[int, int]

class SymbolicScenarioGenerator:

    @staticmethod
    def generate_grid(
        *,
        H: int = 10,
        W: int = 10,
        corridor: bool = False,
        deadend: bool = False,
        min_deadend_depth: int = 3,
        min_corridorLength: int = 3,
        corridor_endpoint_min_free_degree: int = 3,
        z3_seed: Optional[int] = None,
        exact_walls: Optional[int] = None,
        min_walls: Optional[int] = None,
        max_walls: Optional[int] = None,
        spawn: Coord = (1,1),

    ) -> Tuple[
        List[List[bool]],                # grid: True = wall, False = free
        Optional[Coord],                 # chosen deadend endpoint (if any)
        int,                             # deadend depth from start (or -1)
        List[Coord],                     # deadend simple path (ordered)
        List[Coord],                     # corridor simple path (ordered)
    ]:
        """
        Build a 2D grid with optional symbolic dead-end path and corridor path.

        Returns:
            grid           : HxW bools, grid[r][c] == True means WALL.
            chosen_dead    : a chosen dead-end endpoint (if deadend enabled), else None.
            chosen_depth   : distance(start -> chosen_dead) or -1 if none.
            deadend_path   : ordered list of coords along the dead-end chain, may be empty.
            corridor_path  : ordered list of coords along the corridor chain, may be empty.
        """

        BIG = H * W + 10
        sr, sc = spawn
        assert 0 <= sr < H and 0 <= sc < W, "start out of range"

        wall = [[Bool(f"w_{r}_{c}") for c in range(W)] for r in range(H)]
        dist = [[Int(f"d_{r}_{c}") for c in range(W)] for r in range(H)]

        def inb(r: int, c: int) -> bool:
            return 0 <= r < H and 0 <= c < W

        def nbrs(r: int, c: int):
            return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

        s = Solver()
        s.set(timeout=60_000)
        if z3_seed is not None:
            s.set("random_seed", z3_seed)

        # ----- obstacle count constraints -----
        wall_count = Sum(
            If(wall[r][c], 1, 0)
            for r in range(H)
            for c in range(W)
        )
        if exact_walls is not None:
            s.add(wall_count == exact_walls)
        else:
            if min_walls is not None:
                s.add(wall_count >= min_walls)
            if max_walls is not None:
                s.add(wall_count <= max_walls)

        # ----- distance semantics -----
        for r in range(H):
            for c in range(W):
                s.add(
                    If(
                        wall[r][c],
                        dist[r][c] == BIG,
                        And(dist[r][c] >= 0, dist[r][c] <= BIG),
                    )
                )

        # start cell free & dist = 0
        s.add(Not(wall[sr][sc]))
        s.add(dist[sr][sc] == 0)

        # shortest-path style constraints
        for r in range(H):
            for c in range(W):
                if (r, c) == (sr, sc):
                    continue
                nbs = [(nr, nc) for (nr, nc) in nbrs(r, c) if inb(nr, nc)]
                if not nbs:
                    continue

                upper_bounds = []
                witness_eqs = []
                for (nr, nc) in nbs:
                    # free cell can't be more than +1 from any free neighbor
                    upper_bounds.append(
                        Implies(
                            Not(wall[r][c]),
                            Implies(
                                Not(wall[nr][nc]),
                                dist[r][c] <= dist[nr][nc] + 1,
                            ),
                        )
                    )
                    # at least one neighbor with dist-1
                    witness_eqs.append(
                        And(
                            Not(wall[nr][nc]),
                            dist[r][c] == dist[nr][nc] + 1,
                        )
                    )

                s.add(*upper_bounds)
                s.add(Implies(Not(wall[r][c]), Or(*witness_eqs)))

        # ----- dead-end simple path + leaf -----
        if deadend:
            dead_path = [[Bool(f"dp_{r}_{c}") for c in range(W)] for r in range(H)]
            dead_endpoint = [[Bool(f"de_{r}_{c}") for c in range(W)] for r in range(H)]

            dead_endpoint_flags = []
            dead_leaf_endpoint_flags = []
            dead_path_cells = []

            for r in range(H):
                for c in range(W):
                    nbs = [(nr, nc) for (nr, nc) in nbrs(r, c) if inb(nr, nc)]

                    # prevent deadend path cells from lying on the outer boundary of the block
                    # This keeps their free-neighbor degree stable when the procedural stage
                    # carves outside the symbolic block.
                    if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                        s.add(Not(dead_path[r][c]))

                    # degree in the deadend path-subgraph
                    degP = Sum(If(dead_path[nr][nc], 1, 0) for (nr, nc) in nbs)
                    # degree in full free graph
                    degFree = Sum(If(Not(wall[nr][nc]), 1, 0) for (nr, nc) in nbs)

                    # path cells must be free
                    s.add(Implies(dead_path[r][c], Not(wall[r][c])))

                    # chain structure
                    s.add(
                        Implies(
                            dead_path[r][c],
                            Or(
                                And(dead_endpoint[r][c], degP == 1),
                                And(Not(dead_endpoint[r][c]), degP == 2),
                            ),
                        )
                    )

                    # internal dead-path cells: exactly 2 free neighbors
                    s.add(
                        Implies(
                            And(dead_path[r][c], Not(dead_endpoint[r][c])),
                            degFree == 2,
                        )
                    )

                    # off-path cells cannot be endpoints
                    s.add(Implies(Not(dead_path[r][c]), Not(dead_endpoint[r][c])))

                    dead_endpoint_flags.append(If(dead_endpoint[r][c], 1, 0))
                    dead_path_cells.append(If(dead_path[r][c], 1, 0))
                    dead_leaf_endpoint_flags.append(
                        If(
                            And(dead_endpoint[r][c], degFree == 1),
                            1,
                            0,
                        )
                    )

            # exactly two endpoints in the deadend chain
            s.add(Sum(dead_endpoint_flags) == 2)

            # at least one endpoint is a geometric leaf
            s.add(Sum(dead_leaf_endpoint_flags) == 1)

            # strict length: #cells == min_deadend_depth + 1 (edges)
            s.add(Sum(dead_path_cells) == min_deadend_depth + 1)

        # ----- corridor simple path + roomy endpoints -----
        if corridor:
            n_corr = min_corridorLength

            path = [[Bool(f"p_{r}_{c}") for c in range(W)] for r in range(H)]
            endpoint = [[Bool(f"e_{r}_{c}") for c in range(W)] for r in range(H)]

            for r in range(H):
                for c in range(W):
                    s.add(Implies(path[r][c], Not(wall[r][c])))

            endpoints_list = []
            path_cells = []

            for r in range(H):
                for c in range(W):
                    nbs = [(nr, nc) for (nr, nc) in nbrs(r, c) if inb(nr, nc)]

                    # prevent corridor path cells from lying on the outer boundary of the block
                    # for the same reason as the deadend path: we don't want procedural carving
                    # outside the block to change their free-neighbor degree.
                    if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                        s.add(Not(path[r][c]))

                    degP = Sum(If(path[nr][nc], 1, 0) for (nr, nc) in nbs)
                    degFree = Sum(If(Not(wall[nr][nc]), 1, 0) for (nr, nc) in nbs)

                    s.add(
                        Implies(
                            path[r][c],
                            Or(
                                And(endpoint[r][c], degP == 1),
                                And(Not(endpoint[r][c]), degP == 2),
                            ),
                        )
                    )

                    # internal corridor cells: exactly 2 free neighbors
                    s.add(
                        Implies(
                            And(path[r][c], Not(endpoint[r][c])),
                            degFree == 2,
                        )
                    )

                    s.add(Implies(Not(path[r][c]), Not(endpoint[r][c])))

                    # endpoints in "rooms": enough free neighbors
                    s.add(
                        Implies(
                            endpoint[r][c],
                            degFree >= corridor_endpoint_min_free_degree,
                        )
                    )

                    endpoints_list.append(If(endpoint[r][c], 1, 0))
                    path_cells.append(If(path[r][c], 1, 0))

            # exactly one chain: two endpoints
            s.add(Sum(endpoints_list) == 2)

            # strict length: #cells == n_corr + 1
            s.add(Sum(path_cells) == n_corr + 1)

        # ----- solve -----
        t0 = time.time()
        res = s.check()
        print(f"[symbolic] solve_status={res} in {time.time() - t0:.2f}s")

        if str(res) != "sat":
            raise RuntimeError("Unsatisfiable constraints. Relax parameters.")

        m = s.model()
        grid = [[is_true(m.evaluate(wall[r][c])) for c in range(W)] for r in range(H)]

        # ----- extract deadend path -----
        deadend_path: List[Coord] = []
        chosen_dead: Optional[Coord] = None
        chosen_depth: int = -1

        if deadend:
            path_cells_list: List[Coord] = []
            for r in range(H):
                for c in range(W):
                    if is_true(m.evaluate(dead_path[r][c])):
                        path_cells_list.append((r, c))

            if path_cells_list:
                path_set = set(path_cells_list)

                def path_neighbors(rc: Coord) -> List[Coord]:
                    rr, cc = rc
                    return [(nr, nc) for (nr, nc) in nbrs(rr, cc) if (nr, nc) in path_set]

                # endpoints of the deadend path (degree 1 inside the dead_path graph)
                endpoints = [rc for rc in path_cells_list if len(path_neighbors(rc)) == 1]

                def free_degree(rc: Coord) -> int:
                    rr, cc = rc
                    deg = 0
                    for (nr, nc) in nbrs(rr, cc):
                        if inb(nr, nc) and (not is_true(m.evaluate(wall[nr][nc]))):
                            deg += 1
                    return deg

                # the “tip” should be the endpoint that is also a free-graph leaf
                leaf_endpoints = [rc for rc in endpoints if free_degree(rc) == 1]

                # pick start/end deterministically
                start: Optional[Coord] = None
                end: Optional[Coord] = None

                if leaf_endpoints and len(endpoints) >= 2:
                    # if you added ==1 constraint, this should be unique
                    tip = leaf_endpoints[0]
                    other = endpoints[0] if endpoints[0] != tip else endpoints[1]
                    start, end = other, tip

                elif len(endpoints) >= 2:
                    # fallback: use dist to choose farther endpoint as tip
                    e1, e2 = endpoints[0], endpoints[1]
                    dv1 = m.evaluate(dist[e1[0]][e1[1]]).as_long()
                    dv2 = m.evaluate(dist[e2[0]][e2[1]]).as_long()
                    if dv1 <= dv2:
                        start, end = e1, e2
                        chosen_depth = dv2
                    else:
                        start, end = e2, e1
                        chosen_depth = dv1

                # build ordered chain start->end
                if start is not None and end is not None:
                    ordered: List[Coord] = [start]
                    prev: Optional[Coord] = None
                    cur: Coord = start

                    while cur != end:
                        nbs = [nb for nb in path_neighbors(cur) if nb != prev]
                        if not nbs:
                            break
                        nxt = nbs[0]  # deadend path is a chain
                        ordered.append(nxt)
                        prev, cur = cur, nxt

                    deadend_path = ordered

                    # tip is the last cell if we successfully reached end, otherwise still use last
                    chosen_dead = deadend_path[-1]
                    if chosen_depth < 0:
                        chosen_depth = m.evaluate(dist[chosen_dead[0]][chosen_dead[1]]).as_long()
                else:
                    deadend_path = []
                    chosen_dead = None
                    chosen_depth = -1


        # ----- extract corridor path -----
        corridor_path: List[Coord] = []
        if corridor:

            corridor_cells: List[Coord] = []
            for r in range(H):
                for c in range(W):
                    # type: ignore[name-defined]  # path exists if corridor True
                    if is_true(m.evaluate(path[r][c])):
                        corridor_cells.append((r, c))

            if corridor_cells:
                corridor_set = set(corridor_cells)

                def corr_neighbors(rc: Coord) -> List[Coord]:
                    rr, cc = rc
                    return [
                        (nr, nc)
                        for (nr, nc) in nbrs(rr, cc)
                        if (nr, nc) in corridor_set
                    ]

                corr_endpoints = [
                    rc for rc in corridor_cells
                    if len(corr_neighbors(rc)) == 1
                ]

                if corr_endpoints:
                    start = corr_endpoints[0]
                else:
                    start = corridor_cells[0]

                ordered_corr: List[Coord] = [start]
                prev: Optional[Coord] = None
                cur: Coord = start

                while True:
                    nbs = [nb for nb in corr_neighbors(cur) if nb != prev]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    ordered_corr.append(nxt)
                    prev, cur = cur, nxt

                corridor_path = ordered_corr

        return grid, chosen_dead, chosen_depth, deadend_path, corridor_path
