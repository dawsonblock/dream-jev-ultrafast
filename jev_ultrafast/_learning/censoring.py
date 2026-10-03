"""_learning.censoring — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

__all__ = [
    '_run_outcome',
]


def _run_outcome(final: dict | None) -> str:
    """Terminal outcome class of a recorded run.

    ``success`` — verifier-confirmed done. ``failure`` — a measured terminal
    non-success: blocked, denied, budget exhaustion, or a done-claim the
    verifier did not confirm. ``censored`` — no outcome was measured at all:
    the run was aborted, the recording was interrupted before
    ``run_finished``, or it ended ``claimed_done`` with no verifier ever
    checking. Censored is *missing data*, not a negative example — an
    operator closing a session or a browser crash says nothing about whether
    the trajectory was working.
    """
    if final is None:
        return "censored"
    status = str(final.get("status") or "")
    if status == "done" and final.get("verified"):
        return "success"
    if status in {"aborted", "claimed_done"}:
        return "censored"
    return "failure"
