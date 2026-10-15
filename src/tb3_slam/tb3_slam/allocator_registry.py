"""Allocator registry and factory (issue #11).

Maps a short name (used by the coordinator's launch argument) to a concrete
allocator class, so the only difference between the exploration methods is the
allocator selected here. Kept separate from ``allocator.py`` (which holds the
interface) so concrete allocators can import the interface without a cycle.
Classical (#12) and RL (#13) allocators register here as they are added.
"""
from __future__ import annotations

from tb3_slam.allocator import Allocator
from tb3_slam.nearest_frontier_allocator import NearestFrontierAllocator

#: Known allocators by name.
#:
#: ``dummy``/``nearest`` are the issue #11 stub. ``classic`` and ``rl`` are the
#: two user-facing exploration methods: ``exploration_mode:=classic|rl`` maps
#: onto these names so the method difference lives entirely in the allocator
#: and nothing else in the stack changes. The real classical (#12) and RL
#: (#13) allocators replace the stub entries here as they land; until then
#: both fall back to the nearest-frontier dummy so the plumbing runs
#: end to end.
ALLOCATORS: dict[str, type] = {
    'dummy': NearestFrontierAllocator,
    'nearest': NearestFrontierAllocator,
    'classic': NearestFrontierAllocator,  # TODO(#12): cost-utility allocator
    'rl': NearestFrontierAllocator,        # TODO(#13): pointwise-DQN allocator
}


def make_allocator(name: str, **kwargs) -> Allocator:
    """Instantiate a registered allocator by name.

    Raises ``ValueError`` for an unknown name so a bad launch argument fails
    loudly rather than silently falling back.
    """
    try:
        cls = ALLOCATORS[name]
    except KeyError:
        known = ', '.join(sorted(ALLOCATORS))
        raise ValueError(f'unknown allocator {name!r}; known: {known}') from None
    return cls(**kwargs)
