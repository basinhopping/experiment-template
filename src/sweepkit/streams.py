"""Named random streams: every random component draws from its own generator, derived from the run's seed and the
component's name, so a new option never changes the draws of a configuration without it (a new name is a new
stream; existing streams are untouched)."""
import hashlib

import numpy as np


def stream_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def generator(seed: int, name: str) -> np.random.Generator:
    """numpy Generator for one named stream of one seed."""
    return np.random.default_rng([int(seed), stream_key(name)])


def int_seed(seed: int, name: str) -> int:
    """A 31-bit integer seed for one named stream (for libraries that take an int random_state)."""
    return int(np.random.SeedSequence([int(seed), stream_key(name)]).generate_state(1, dtype=np.uint32)[0] >> 1)
