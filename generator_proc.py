from __future__ import annotations
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import random
import os

Coord = Tuple[int, int]


@dataclass
class GridSpec:
    H: int = 20
    W: int = 20


class ProceduralScenarioGenerator:
    def __init__(self, grid: GridSpec):
        self.grid = grid

    @staticmethod
    def _nbrs(h: int, w: int, r: int, c: int):
        if r > 0:
            yield r - 1, c
        if r + 1 < h:
            yield r + 1, c
        if c > 0:
            yield r, c - 1
        if c + 1 < w:
            yield r, c + 1

    def bfs_dist(self, grid: List[List[bool]], root: Coord):
        H, W = self.grid.H, self.grid.W
        sr, sc = root
        if not (0 <= sr < H and 0 <= sc < W) or grid[sr][sc]:
            return None, set()

        dist = [[-1] * W for _ in range(H)]
        q = deque([(sr, sc)])
        dist[sr][sc] = 0
        seen = {(sr, sc)}

        while q:
            r, c = q.popleft()
            for nr, nc in self._nbrs(H, W, r, c):
                if not grid[nr][nc] and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    seen.add((nr, nc))
                    q.append((nr, nc))

        return dist, seen

    @staticmethod
    def deg_free(grid: List[List[bool]], r: int, c: int) -> int:
        if grid[r][c]:
            return 0
        H, W = len(grid), len(grid[0])
        d = 0
        if r > 0 and not grid[r - 1][c]:
            d += 1
        if r + 1 < H and not grid[r + 1][c]:
            d += 1
        if c > 0 and not grid[r][c - 1]:
            d += 1
        if c + 1 < W and not grid[r][c + 1]:
            d += 1
        return d

    def count_deadends(self, grid: List[List[bool]], root: Coord, min_depth: int):
        dist, seen = self.bfs_dist(grid, root)
        if dist is None:
            return 0, []
        hits = []
        for (r, c) in seen:
            if self.deg_free(grid, r, c) == 1 and dist[r][c] >= min_depth:
                hits.append((r, c))
        return len(hits), hits

    def max_corridor_length(self, grid: List[List[bool]], root: Coord) -> int:
        dist, seen = self.bfs_dist(grid, root)
        if dist is None or not seen:
            return 0
        a = max(seen, key=lambda rc: dist[rc[0]][rc[1]])
        dist_a, seen_a = self.bfs_dist(grid, a)
        if dist_a is None:
            return 0
        b = max(seen_a, key=lambda rc: dist_a[rc[0]][rc[1]])
        return dist_a[b[0]][b[1]]

    # ---------- Procedural generation ----------
    def generate_connected_grid_proc(
        self,
        root: Coord = (1, 1),
        exact_walls: Optional[int] = None,
        min_walls: Optional[int] = None,
        max_walls: Optional[int] = None,
        rng: Optional[random.Random] = None,
        base_grid: Optional[List[List[bool]]] = None,
        frozen: Optional[Set[Coord]] = None,
    ) -> List[List[bool]]:
        H, W = self.grid.H, self.grid.W
        sr, sc = root
        assert 0 <= sr < H and 0 <= sc < W, "root out of range"

        rng = rng or random.Random()
        frozen = frozen or set()
        total = H * W

        # choose target wall count
        if exact_walls is not None:
            target_walls = exact_walls
        else:
            lo = min_walls if min_walls is not None else 0
            hi = max_walls if max_walls is not None else total
            if lo > hi:
                raise ValueError("min_walls > max_walls")
            target_walls = rng.randint(lo, hi)

        # initialize grid
        if base_grid is not None:
            assert len(base_grid) == H and len(base_grid[0]) == W, "base_grid shape mismatch"
            grid = [row[:] for row in base_grid]

            if grid[sr][sc]:
                if (sr, sc) in frozen:
                    raise ValueError("Root is in a frozen wall cell")
                grid[sr][sc] = False

            free_set: Set[Coord] = set()
            for r in range(H):
                for c in range(W):
                    if not grid[r][c]:
                        free_set.add((r, c))

            frontier: Set[Coord] = set()
            for (r, c) in free_set:
                for nr, nc in self._nbrs(H, W, r, c):
                    if grid[nr][nc] and (nr, nc) not in frozen:
                        frontier.add((nr, nc))
        else:
            grid = [[True for _ in range(W)] for _ in range(H)]
            free_set: Set[Coord] = set()
            frontier: Set[Coord] = set()

        def add_free(cell: Coord):
            r, c = cell
            if (r, c) in frozen:
                return
            if (r, c) in free_set:
                return
            grid[r][c] = False
            free_set.add((r, c))
            for nr, nc in self._nbrs(H, W, r, c):
                if (nr, nc) not in free_set and (nr, nc) not in frozen:
                    frontier.add((nr, nc))

        if base_grid is None:
            add_free(root)

        base_free = len(free_set)
        target_free_raw = total - target_walls
        target_free_raw = max(1, min(total, target_free_raw))
        target_free = max(base_free, target_free_raw)

        # carve with frontier
        while len(free_set) < target_free and frontier:
            r, c = rng.choice(tuple(frontier))
            frontier.discard((r, c))
            if (r, c) in frozen:
                continue
            neighbors = sum(
                1 for nr, nc in self._nbrs(H, W, r, c)
                if (nr, nc) in free_set
            )
            if neighbors == 1 or rng.random() < 0.25:
                add_free((r, c))

        # fill if still short
        while len(free_set) < target_free:
            candidates = []
            for (r, c) in list(free_set):
                for nr, nc in self._nbrs(H, W, r, c):
                    if (nr, nc) in free_set or (nr, nc) in frozen:
                        continue
                    candidates.append((nr, nc))
            if not candidates:
                break
            add_free(rng.choice(candidates))

        print("DEBUG: final free_set size:", len(free_set))

        return grid

    # ---------- One-call generator with guarantees (BFS-checked) ----------
    def generate_with_requirements(
        self,
        root: Coord,
        min_walls: Optional[int] = None,
        max_walls: Optional[int] = None,
        min_corridor: Optional[int] = None,
        min_deadends: Optional[int] = None,
        min_deadend_depth: int = 3,
        rng: Optional[random.Random] = None,
        max_tries: int = 200,
        base_grid: Optional[List[List[bool]]] = None,
        frozen: Optional[Set[Coord]] = None,
    ) -> List[List[bool]]:
        rng = rng or random.Random()
        for _ in range(max_tries):
            grid = self.generate_connected_grid_proc(
                root=root,
                min_walls=min_walls,
                max_walls=max_walls,
                rng=rng,
                base_grid=base_grid,
                frozen=frozen,
            )
            # if not self.py_check_connected(grid, root):
            #     continue
            if min_corridor is not None:
                if self.max_corridor_length(grid, root) < min_corridor:
                    continue
            if min_deadends is not None:
                cnt, _ = self.count_deadends(grid, root, min_deadend_depth)
                if cnt < min_deadends:
                    continue
            return grid
        raise RuntimeError("Could not satisfy requirements within max_tries")

    # ---------- MuJoCo XML ----------
    def grid_to_mujoco_xml_base_compatible(
        self,
        grid: List[List[bool]],
        model_name: str = "simple_navigation",
        arena_half_extent: float = 20.0,
        height: float = 0.25,
        agent_cells: Optional[List[Coord]] = None,
        target_cells: Optional[List[Coord]] = None,
    ) -> str:
        H, W = self.grid.H, self.grid.W
        A = float(arena_half_extent)
        cell_w = (2.0 * A) / W # cell size = 2x2
        cell_h = (2.0 * A) / H
        half_w, half_h = cell_w * 0.40, cell_h * 0.40 # obstacle size = 2 * 2 * 0.4 = 1.6

        agent_cells = agent_cells or []
        target_cells = target_cells or []

        def cell_to_xy(r: int, c: int):
            x = -A + (c + 0.5) * cell_w
            y = A - (r + 0.5) * cell_h
            return x, y

        obstacles_xml = []
        for r in range(H):
            for c in range(W):
                if not grid[r][c]:
                    continue
                x, y = cell_to_xy(r, c)
                obstacles_xml.append(
                    f'<body name="ob_{r}_{c}" pos="{x:.4f} {y:.4f} {height:.3f}">'
                    f'  <geom type="box" size="{half_w:.4f} {half_h:.4f} {height:.3f}" '
                    f'        rgba="1 0 0 1" contype="1" conaffinity="1"/>'
                    f'</body>'
                )
        obstacles_text = "\n        ".join(obstacles_xml)

        agent_bodies = []
        for i, (r, c) in enumerate(agent_cells): # agents are 1.5 x 1.5
            x, y = cell_to_xy(r, c)
            agent_bodies.append(f"""
            <body name="agent_{i}" pos="{x:.4f} {y:.4f} {height:.3f}">
                <joint name="agent_{i}_j1" type="slide" axis="1 0 0"/>
                <joint name="agent_{i}_j2" type="slide" axis="0 1 0"/>
                <geom  name="agent_geom_{i}" type="box" size="0.25 0.25 {height:.3f}" rgba="1 1 0 1"/> 
            </body>
            """)

        target_bodies = []
        for j, (r, c) in enumerate(target_cells):
            x, y = cell_to_xy(r, c)
            target_bodies.append(f"""
            <body name="target_{j}" pos="{x:.4f} {y:.4f} 0.25">
                <geom name="target_geom_{j}" type="sphere" size="0.25"
                      rgba="0 1 0 1" contype="0" conaffinity="0"/>
            </body>
            """)

        agents_text = "\n".join(agent_bodies + target_bodies)

        return f"""<mujoco model="{model_name}">
  <option timestep="0.01"/>

  <worldbody>
    <geom name="floor" type="plane" size="{A} {A} 0.1" rgba="0.5 0.5 0.5 1"/>

    {obstacles_text}
    {agents_text}
  </worldbody>

</mujoco>"""

    @staticmethod
    def write_xml(xml: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)