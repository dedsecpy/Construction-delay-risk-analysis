"""Asymmetric alert utility: miss severe costs 5x a false alarm."""

from __future__ import annotations

MISS_COST = 5.0
FA_COST = 1.0


def expected_alert_utility(p_severe: float, threshold: float = 0.5) -> float:
    """Utility if we alert when p_severe >= threshold (negative costs)."""
    # Two actions: alert / no-alert. Choose best expected utility.
    # alert: FA_COST if not severe; 0 if severe (caught)
    # no-alert: MISS_COST if severe; 0 if not
    u_alert = -(1 - p_severe) * FA_COST
    u_quiet = -p_severe * MISS_COST
    return max(u_alert, u_quiet)


def should_alert(p_severe: float, threshold: float = 0.65) -> bool:
    """Operating-point alert used for scoreboard metrics (calibrated threshold).

    VoI selection still uses asymmetric expected_alert_utility; the final yes/no
    alert for evaluation uses an explicit probability threshold (default 0.5).
    """
    return float(p_severe) >= float(threshold)
