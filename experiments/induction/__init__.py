# experiments/induction/__init__.py
"""experiments/induction/__init__.py
Repeated-random induction behavior and mechanistic DAC probes.
"""

from .behavior import (
    DEFAULT_MAX_FORWARD_TOKENS,
    InductionBehaviorResult,
    auto_micro_batch_size,
    evaluate_induction_behavior,
)

from .dac import (
    DEFAULT_OFFSETS,
    DACInductionCollector,
    top_heads,
)


__all__ = [
    "DEFAULT_MAX_FORWARD_TOKENS",
    "DEFAULT_OFFSETS",
    "InductionBehaviorResult",
    "DACInductionCollector",
    "auto_micro_batch_size",
    "evaluate_induction_behavior",
    "top_heads",
]
