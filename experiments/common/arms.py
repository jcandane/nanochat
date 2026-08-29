"""experiments/common/arms.py
Canonical attention-arm definitions for experiments.

The operator square itself lives in :mod:`nanochat.gpt`. Experiments must import
arm coordinates through this module instead of re-declaring ``(beta, alpha)``
tuples in individual probes, Modal launchers, or evaluation scripts.

Coordinate convention
---------------------
``beta`` releases the kinetic-signature constraint and ``alpha`` releases the
exact/Doob constraint into flux::

    DMAP      (beta=0, alpha=0)  PSD kinetic       + Doob/exact
    AMAP      (beta=0, alpha=1)  PSD kinetic       + flux
    HMAP      (beta=1, alpha=0)  indefinite kinetic + Doob/exact
    attention (beta=1, alpha=1)  indefinite kinetic + flux

The ``attention`` arm normally uses nanochat's standard FA3/SDPA backend. Its
``(1, 1)`` coordinates are retained for provenance and for eager-equivalence
tests of the operator-square implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from nanochat.gpt import HMAP_ARMS as _CANONICAL_ARMS


ArmName = Literal["attention", "amap", "hmap", "dmap"]

# Preferred presentation / sweep order. Coordinates remain owned by nanochat.gpt.
ARM_NAMES: tuple[ArmName, ...] = ("attention", "amap", "hmap", "dmap")

# Fail loudly if nanochat.gpt and the experiment API ever drift apart.
if set(ARM_NAMES) != set(_CANONICAL_ARMS):
    raise RuntimeError(
        "experiments.common.arms is out of sync with nanochat.gpt.HMAP_ARMS: "
        f"experiments={sorted(ARM_NAMES)}, gpt={sorted(_CANONICAL_ARMS)}"
    )


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Resolved operator arm used by training/evaluation experiments."""

    name: ArmName
    beta: float
    alpha: float
    attn_variant: Literal["standard", "hmap"]

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return ``(beta, alpha)`` in operator-square order."""
        return self.beta, self.alpha

    @property
    def hf_folder(self) -> str:
        """Canonical top-level folder name in the shared Hugging Face repo."""
        return "attention" if self.name == "attention" else self.name.upper()

    @property
    def active_flux_coefficient(self) -> float:
        """Coefficient multiplying the antisymmetric q/k flux sector.

        Under the current constraint-release convention this is ``alpha`` for
        the eager operator family. Standard attention is the fully released
        ``(1, 1)`` corner and therefore also has coefficient 1.
        """
        return 1.0 if self.attn_variant == "standard" else self.alpha

    @property
    def exact_potential_coefficient(self) -> float:
        """Coefficient multiplying the Doob/exact potential sector."""
        return 0.0 if self.attn_variant == "standard" else 1.0 - self.alpha

    @property
    def nsd_coefficient(self) -> float:
        """Coefficient of the negative-semidefinite kinetic sector."""
        return self.beta

    def config_overrides(self) -> dict[str, object]:
        """Keyword overrides for :class:`nanochat.gpt.GPTConfig`.

        Keeping alpha/beta populated for standard attention is intentional: the
        standard backend ignores them, while metadata still records attention
        at the ``(1, 1)`` corner.
        """
        return {
            "attn_variant": self.attn_variant,
            "hmap_beta": self.beta,
            "hmap_alpha": self.alpha,
        }

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly representation for provenance and result files."""
        return {
            "name": self.name,
            "hf_folder": self.hf_folder,
            "beta": self.beta,
            "alpha": self.alpha,
            "attn_variant": self.attn_variant,
            "active_flux_coefficient": self.active_flux_coefficient,
            "exact_potential_coefficient": self.exact_potential_coefficient,
            "nsd_coefficient": self.nsd_coefficient,
        }


def normalize_arm_name(name: str) -> ArmName:
    """Normalize a CLI/user arm name and validate it."""
    normalized = name.strip().lower()
    if normalized not in _CANONICAL_ARMS:
        choices = ", ".join(ARM_NAMES)
        raise ValueError(f"unknown attention arm {name!r}; expected one of: {choices}")
    return normalized  # type: ignore[return-value]


def get_arm(name: str) -> ArmSpec:
    """Resolve an arm from the single canonical square in ``nanochat.gpt``."""
    arm_name = normalize_arm_name(name)
    beta, alpha = _CANONICAL_ARMS[arm_name]
    return ArmSpec(
        name=arm_name,
        beta=float(beta),
        alpha=float(alpha),
        attn_variant="standard" if arm_name == "attention" else "hmap",
    )


def iter_arms() -> Iterator[ArmSpec]:
    """Iterate over all four arms in stable experiment/reporting order."""
    for name in ARM_NAMES:
        yield get_arm(name)
