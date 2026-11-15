"""MTAH team pipeline package.

Decoder symbols are loaded lazily so ``python -m mtah.decode`` can execute without importing the
module once through the package initializer first.
"""

from importlib import import_module

__all__ = [
    "H",
    "A",
    "D",
    "LETTERS",
    "viterbi",
    "viterbi_min_duration",
    "viterbi_restart_bonus",
    "decode",
    "probabilities_to_logp",
    "restart_transition_bonus",
    "snap_restart_boundaries",
    "merge_short_live_spells",
    "decode_probabilities",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".decode", __name__), name)
