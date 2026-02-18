from tb3_exploration.frontier_utils import find_centroid, find_frontiers, find_groups, find_nearest_frontier


def normalize(groups):
    """Help to make group comparison order-independent."""
    return sorted(
        [sorted(group) for group in groups],
        key=lambda g: (len(g), g)
    )


def test_find_frontiers_basic():
    data = [
        [-1, -1,  0, 0],
        [-1,  0,  0, 0],
        [-1, -1, -1, 0]
    ]
    expected = [(0, 1), (1, 0), (2, 1), (2, 2)]
    result = find_frontiers(data)
    assert set(result) == set(expected)


def test_find_groups_basic():
    points = [
        (0, 0), (1, 0), (1, 1),
        (5, 5), (6, 6), (7, 7),
        (10, 0)
    ]
    expected = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (7, 7)],
        [(10, 0)]
    ]
    result = find_groups(points)
    assert normalize(result) == normalize(expected)


def test_find_centroid():
    frontier = [(0, 1), (1, 0), (2, 1), (2, 2)]
    expected = (1, 1)
    result = find_centroid(frontier)
    assert result == expected


def test_find_nearest_frontier():
    centroid = (0, 3)
    frontier = [(0, 1), (1, 0), (2, 1), (2, 2)]
    expected = (0, 1)
    result = find_nearest_frontier(centroid, frontier)
    assert result == expected
