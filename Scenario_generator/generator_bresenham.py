from __future__ import annotations
from typing import List, Tuple, Optional
from collections import deque
import random

Coord = Tuple[int, int]

class BresenhamStandardGenerator:
    def __init__(self, H: int, W: int):
        self.H = H
        self.W = W

    def bresenham_line(self, start: Coord, end: Coord) -> list[Coord]:
        """Return list of (r, c) from start to end."""
        (x0, y0) = start
        (x1, y1) = end
        points = []

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        return points

    def generate(
        self,
        *,
        agent: Coord,
        target: Coord,
        exact_walls: Optional[int],
        min_walls: Optional[int],
        max_walls: Optional[int],
        line_wall_fraction: float = 0.7,
        neighbor_radius: int = 1,
        rng: random.Random,
    ) -> list[list[bool]]:
        """
        True = WALL, False = FREE
        Place most walls near the Bresenham line between agent and target.
        `neighbor_radius` controls how thick the band around the line is.
        """
        H, W = self.H, self.W

        # --- 1) Decide total number of walls ---
        total_cells = H * W
        if exact_walls is not None:
            num_walls = exact_walls
        else:
            lo = min_walls if min_walls is not None else 0
            hi = max_walls if max_walls is not None else total_cells
            if lo > hi:
                raise ValueError("min_walls cannot be > max_walls")
            num_walls = rng.randint(lo, hi)

        num_walls = max(0, min(num_walls, total_cells - 2))  # keep some free for agent/target

        # --- 2) Initialize grid as all free ---
        grid = [[False for _ in range(W)] for _ in range(H)]

        # --- 3) Compute Bresenham line ---
        line_cells = self.bresenham_line(agent, target)

        # --- 4) Build a "band" around the line (line + neighbors) ---
        band_cells: set[Coord] = set()

        # don't ever wall agent/target
        forbidden = {agent, target}

        for (r, c) in line_cells:
            # skip agent/target explicitly; they must stay free
            if (r, c) not in forbidden:
                band_cells.add((r, c))

            if neighbor_radius > 0:
                for dr in range(-neighbor_radius, neighbor_radius + 1):
                    for dc in range(-neighbor_radius, neighbor_radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            if (rr, cc) not in forbidden:
                                band_cells.add((rr, cc))

        band_cells = list(band_cells)
        rng.shuffle(band_cells)

        # --- 5) Place walls, prioritizing the band ---
        walls_left = num_walls

        # 5a) walls in band
        band_to_use = min(walls_left, len(band_cells))
        for i in range(band_to_use):
            (r, c) = band_cells[i]
            grid[r][c] = True
        walls_left -= band_to_use

        if walls_left > 0:
            # 5b) remaining walls anywhere else (excluding agent/target and existing walls)
            other_candidates = [
                (r, c)
                for r in range(H)
                for c in range(W)
                if not grid[r][c] and (r, c) not in forbidden
            ]
            rng.shuffle(other_candidates)
            extra_to_use = min(walls_left, len(other_candidates))
            for i in range(extra_to_use):
                (r, c) = other_candidates[i]
                grid[r][c] = True

        return grid

