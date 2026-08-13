"""Shared-CPI summary-only ablation with every curvature feature removed.

This runner deliberately reuses the same simulated trials as CORE3.  Its
behavioral input contains the non-curvature summary statistics only, making it
a matched control for testing whether the remaining turn features help or hurt
human CPI ordering.
"""

from __future__ import annotations

from . import shared_cpi_2step_run as runner


runner.FEATURE_PROFILES["speedpath"] = tuple(runner.CURVATURE_FEATURES)


if __name__ == "__main__":
    raise SystemExit(runner.main())
