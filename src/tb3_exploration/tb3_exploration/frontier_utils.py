import math

import numpy as np


def find_frontiers(data) -> list[tuple[int, int]]:
    """
    Find indices of 'frontier' cells in a 2D matrix.

    Frontier: Cell with -1, that neighbors at least one 0 (free cell).

    Args:
        data (np.ndarray): 2D matrix of integers.

    Returns:
        frontiers: list[tuple[int, int]]: List of (row, col)-indices of the frontiers.
    """
    data = np.array(data)
    frontiers = []
    rows, cols = data.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(rows):
        for j in range(cols):
            if data[i, j] == -1:
                for dr, dc in neighbors:
                    nr, nc = i + dr, j + dc
                    if 0 <= nr < rows and 0 <= nc < cols and 0 <= data[nr, nc] <= 25:  # 25 widens range of possible free cells (instead of using 0)
                        frontiers.append((i, j))
                        break
    return frontiers


def find_groups(frontiers: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """
    Find connected groups of frontiers.

    Args:
        frontiers (list[tuple[int, int]]): list of coordinates of frontiers

    Returns:
        list[list[tuple[int, int]]]: list of groups of frontiers
    """
    point_set = set(frontiers)
    visited = set()
    groups = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    for p in point_set:
        if p in visited:
            continue

        stack = [p]
        visited.add(p)
        group = [p]

        while stack:
            x, y = stack.pop()
            for dx, dy in neighbors:
                q = (x + dx, y + dy)
                if q in point_set and q not in visited:
                    visited.add(q)
                    stack.append(q)
                    group.append(q)

        groups.append(group)

    return groups


def find_centroid(frontier: list[tuple[int, int]], grid: np.ndarray) -> tuple[int, int] | None:
    """Compute the centroid of a frontier and return a navigable free cell."""
    if not frontier:
        return None

    # 1) Berechne Schwerpunkt
    n = len(frontier)
    sum_row = sum(r for r, _c in frontier)
    sum_col = sum(c for _r, c in frontier)
    centroid = (sum_row // n, sum_col // n)  # (row, col)

    rows, cols = grid.shape

    # 2) Prüfe, ob centroid auf free (0) liegt
    r, c = centroid
    if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
        return centroid

    # 3) Finde nächstgelegene Zelle in Frontier
    nearest = frontier[0]
    min_dist = math.hypot(r - nearest[0], c - nearest[1])
    for fr, fc in frontier:
        d = math.hypot(r - fr, c - fc)
        if d < min_dist:
            min_dist = d
            nearest = (fr, fc)

    # 4) Suche einen freien Nachbarn
    fr, fc = nearest
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in neighbors:
        nr, nc = fr + dr, fc + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            return (nr, nc)

    # 5) kein freier Nachbar gefunden
    return None
