"""_learning.confidence — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

import math
from statistics import NormalDist

__all__ = [
    '_SEQUENTIAL_SPEND',
    '_ZETA_1_5',
    '_arm_confidence_bounds',
    '_arm_ws',
    '_arm_wsq',
    '_arm_wsum',
    '_bernoulli_kl',
    '_kl_bound',
    '_newcombe',
    '_sequential_alpha',
    '_wilson_interval',
    '_z_for_alpha',
]


def _wilson_interval(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on a (possibly IPW-weighted) proportion.

    ``n`` is the effective sample size — the count of independent
    observations the weight mass is worth — so wide intervals on thin support
    are intrinsic to the estimate, not a reporting choice.
    """
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


_ZETA_1_5 = 2.6123753486854882


_SEQUENTIAL_SPEND = 1.0 / _ZETA_1_5


def _sequential_alpha(looks: float, *, base: float) -> float:
    """Error budget spent at look ``n``: a summable n^-3/2 spending sequence.

    A contrast is recomputed whenever evidence arrives — an unbounded number
    of peeks — so the per-look budget must sum over all integer sample sizes:
    ``α_n = base·n^-1.5/ζ(3/2)`` spends at most ``base`` in total. Feeding
    this budget into *exact* per-look concentration bounds (KL/Chernoff for
    uniform weights, Hoeffding for non-uniform) composes a formally
    anytime-valid confidence sequence: the probability that the true
    candidate−control difference ever leaves the reported interval is ≤ the
    spent budget, uniformly over all sample sizes. Multiplicity across the
    simultaneously tracked hypothesis family is Bonferroni-controlled —
    ``CounterfactualTrials.SEQUENTIAL_ALPHA`` is split across contexts before
    the look-wise spending is applied.
    """
    n = max(1.0, float(looks))
    return min(0.5, float(base) * _SEQUENTIAL_SPEND / (n ** 1.5))


def _z_for_alpha(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)


def _bernoulli_kl(p: float, q: float) -> float:
    """kl(Bern(p) ‖ Bern(q)) with the standard 0·log 0 = 0 conventions."""
    if p <= 0.0:
        return -math.log1p(-q) if q < 1.0 else math.inf
    if p >= 1.0:
        return -math.log(q) if q > 0.0 else math.inf
    if q <= 0.0 or q >= 1.0:
        return math.inf
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _kl_bound(p_hat: float, n: int, tau: float, *, upper: bool) -> float:
    """One side of ``{p : n·kl(p̂‖p) ≤ τ}`` — the Chernoff/KL confidence bound.

    For iid Bernoulli(p) samples, P_p(p outside the one-sided bound) ≤
    e^{-τ} at *every* n — an exact per-look guarantee, not an asymptotic
    interval. Solved by bisection; degenerate p̂ at the boundary has a
    closed form.
    """
    if n <= 0:
        return 1.0 if upper else 0.0
    if p_hat <= 0.0:
        return 1.0 - math.exp(-tau / n) if upper else 0.0
    if p_hat >= 1.0:
        return 1.0 if upper else math.exp(-tau / n)
    target = tau / n
    lo, hi = (p_hat, 1.0) if upper else (0.0, p_hat)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        inside = _bernoulli_kl(p_hat, mid) <= target
        if upper:
            if inside:
                lo = mid
            else:
                hi = mid
        else:
            if inside:
                hi = mid
            else:
                lo = mid
    return lo if upper else hi


def _arm_confidence_bounds(
    analyzed: int, wsum: float, wsq: float, ws: float, tau: float
) -> tuple[float, float]:
    """Two-sided bound on an arm's success rate, exact at tail budget e^{-τ}.

    IPW weighting is a reweighting of iid Bernoulli outcomes under randomized
    assignment — the recorded propensity decides *which* arm a run lands in,
    not what the arm does. Uniform weights recover the exact Chernoff/KL
    bound on the sample mean over ``analyzed`` draws; non-uniform weights
    fall back to Hoeffding's inequality for a weighted sum of [0,1] outcomes
    (still exact, just wider). Coverage holds at every look simultaneously
    via the spending sequence in ``_contrast``.
    """
    if analyzed <= 0 or wsum <= 0:
        return 0.0, 1.0
    p_hat = ws / wsum
    # All-equal weights ⇒ wsq == n·w² and wsum == n·w ⇒ wsq·analyzed == wsum².
    uniform = abs(wsq * analyzed - wsum * wsum) <= 1e-9 * max(1.0, wsum * wsum)
    if uniform:
        return (
            _kl_bound(p_hat, int(analyzed), tau, upper=False),
            _kl_bound(p_hat, int(analyzed), tau, upper=True),
        )
    radius = math.sqrt(max(0.0, tau * wsq / (2.0 * wsum * wsum)))
    return max(0.0, p_hat - radius), min(1.0, p_hat + radius)


def _arm_ws(counts) -> float:
    return counts[4]


def _arm_wsum(counts) -> float:
    return counts[5]


def _arm_wsq(counts) -> float:
    return counts[6]


def _newcombe(candidate: dict, control: dict, *, z: float = 1.96) -> list[float]:
    """Newcombe-Wilson interval on the difference of two arm proportions.

    Built from the per-arm Wilson bounds at ``z``. The naive normal-approx SE
    collapses to zero when either arm sits at the 0/1 boundary — exactly the
    extreme evidence where a degenerate interval would claim false precision.
    """
    pc, pk = candidate["p_success"], control["p_success"]
    delta = pc - pk
    lc, uc = _wilson_interval(pc, candidate["ess"], z)
    lk, uk = _wilson_interval(pk, control["ess"], z)
    return [
        delta - math.sqrt((pc - lc) ** 2 + (uk - pk) ** 2),
        delta + math.sqrt((uc - pc) ** 2 + (pk - lk) ** 2),
    ]
