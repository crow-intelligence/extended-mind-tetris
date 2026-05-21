"""Rotation math for tetromino cell offsets.

The Tetris environment and its piece module both rely on these pure helpers to
rotate sets of ``(row, col)`` cell offsets clockwise around the origin and to
enumerate the distinct orientations a shape admits. Functions are exercised by
doctests so they participate in the standard ``pytest --doctest-modules`` run.
"""

from __future__ import annotations


def rotate_cw(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Rotate cell offsets 90 degrees clockwise around the origin.

    The result is normalized so the top-left corner of the rotated shape's
    bounding box sits at ``(0, 0)`` and cells are returned in row-major
    sorted order. Two shapes that differ only by translation therefore
    compare equal as lists.

    Args:
        cells: Cell offsets to rotate.

    Returns:
        Rotated cell offsets, sorted and normalized.

    Examples:
        A horizontal I-piece becomes vertical:

        >>> rotate_cw([(0, 0), (0, 1), (0, 2), (0, 3)])
        [(0, 0), (1, 0), (2, 0), (3, 0)]

        Rotating a T-up twice flips it to a T-down:

        >>> rotate_cw(rotate_cw([(0, 1), (1, 0), (1, 1), (1, 2)]))
        [(0, 0), (0, 1), (0, 2), (1, 1)]

        A single cell is invariant under rotation:

        >>> rotate_cw([(0, 0)])
        [(0, 0)]
    """
    rotated = [(c, -r) for r, c in cells]
    min_r = min(r for r, _ in rotated)
    min_c = min(c for _, c in rotated)
    return sorted((r - min_r, c - min_c) for r, c in rotated)


def all_orientations(
    base: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate the distinct orientations of a shape under clockwise rotation.

    Args:
        base: Canonical orientation as cell offsets.

    Returns:
        Distinct orientations in the order they first appear when rotating
        clockwise from ``base``. The list has length 1, 2 or 4 for tetrominos.

    Examples:
        The O-piece has a single orientation:

        >>> all_orientations([(0, 0), (0, 1), (1, 0), (1, 1)])
        [((0, 0), (0, 1), (1, 0), (1, 1))]

        The I-piece has two:

        >>> len(all_orientations([(0, 0), (0, 1), (0, 2), (0, 3)]))
        2

        The T-piece has four:

        >>> len(all_orientations([(0, 1), (1, 0), (1, 1), (1, 2)]))
        4
    """
    seen: list[tuple[tuple[int, int], ...]] = []
    cells = sorted(base)
    for _ in range(4):
        as_tuple = tuple(cells)
        if as_tuple not in seen:
            seen.append(as_tuple)
        cells = rotate_cw(cells)
    return seen
