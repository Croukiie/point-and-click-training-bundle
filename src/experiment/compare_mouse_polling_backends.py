"""Compare standard and buffered Windows Raw Input mouse acquisition."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from .check_mouse_polling import _summarize_events, _write_events_csv
from .raw_input_buffered_win32 import BufferedRawInputMouse
from .raw_input_win32 import RawInputMouse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _capture(label: str, mouse_class, seconds: float):
    print(f"\n[{label.upper()} BACKEND]")
    print("Move the mouse continuously in large circles. Do not click.")
    mouse = mouse_class()
    mouse.start()
    events = []
    try:
        for remaining in range(3, 0, -1):
            print(f"Starting in {remaining}...", flush=True)
            time.sleep(1.0)
        mouse.drain()
        print(f"CAPTURING for {seconds:g} seconds...", flush=True)
        start = time.perf_counter()
        deadline = start + seconds
        while time.perf_counter() < deadline:
            events.extend(mouse.drain())
            time.sleep(0.001)
        events.extend(mouse.drain())
        elapsed = time.perf_counter() - start
    finally:
        mouse.stop()
    return events, elapsed


def _method_summary(label: str, events, elapsed: float, expected_hz: float) -> dict:
    summary = _summarize_events(events, elapsed, expected_hz)
    motion_count = int(summary["motion_events"])
    count_rate = motion_count / elapsed if elapsed > 0 else 0.0
    summary.update(
        {
            "capture_method": label,
            "motion_event_count_rate_hz": round(count_rate, 3),
            "comparison_rate_hz": round(count_rate, 3),
            "timestamp_interpretation": (
                "per-WM_INPUT processing timestamp"
                if label == "standard"
                else "one processing timestamp per GetRawInputBuffer batch; "
                "use event-count rate, not median interval"
            ),
        }
    )
    if label == "buffered":
        summary["effective_rate_from_median_interval_hz"] = None
        summary["interval_median_ms"] = None
        summary["nearest_standard_polling_rate_hz"] = None
        summary["compatible_with_expected_rate"] = count_rate >= expected_hz * 0.75
    return summary


def _verdict(standard_rate: float, buffered_rate: float, expected_hz: float) -> str:
    standard_ok = standard_rate >= expected_hz * 0.75
    buffered_ok = buffered_rate >= expected_hz * 0.75
    relative_difference = abs(buffered_rate - standard_rate) / max(
        standard_rate, buffered_rate, 1.0
    )
    if buffered_ok and not standard_ok:
        return "STANDARD_BACKEND_MISSES_HIGH_FREQUENCY_EVENTS"
    if not standard_ok and not buffered_ok and relative_difference <= 0.15:
        return "BOTH_BACKENDS_AGREE_BELOW_EXPECTED_RATE"
    if standard_ok and buffered_ok and relative_difference <= 0.15:
        return "BOTH_BACKENDS_AGREE_WITH_EXPECTED_RATE"
    return "BACKENDS_DISAGREE_REVIEW_RAW_OUTPUT"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential standard and GetRawInputBuffer mouse captures to "
            "test whether the standard Python logger misses events."
        )
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--expected-hz", type=float, default=1000.0)
    parser.add_argument(
        "--order",
        choices=("standard-first", "buffered-first"),
        default="standard-first",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "diagnostics"
            / "mouse_polling_backend_comparison"
        ),
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.expected_hz <= 0:
        parser.error("--expected-hz must be positive")

    methods = {
        "standard": RawInputMouse,
        "buffered": BufferedRawInputMouse,
    }
    order = (
        ("standard", "buffered")
        if args.order == "standard-first"
        else ("buffered", "standard")
    )
    session_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    summaries = {}

    for label in order:
        events, elapsed = _capture(label, methods[label], args.seconds)
        summary = _method_summary(label, events, elapsed, args.expected_hz)
        summaries[label] = summary
        method_dir = session_dir / label
        method_dir.mkdir()
        _write_events_csv(method_dir / "raw_input_events.csv", events)
        with (method_dir / "polling_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(
            f"{label}: {summary['motion_events']} motion events, "
            f"count rate {summary['motion_event_count_rate_hz']:.1f} Hz"
        )
        if label == "standard":
            median_rate = summary["effective_rate_from_median_interval_hz"]
            if median_rate is not None:
                print(f"standard median-interval rate: {median_rate:.1f} Hz")

    standard_rate = float(summaries["standard"]["comparison_rate_hz"])
    buffered_rate = float(summaries["buffered"]["comparison_rate_hz"])
    verdict = _verdict(standard_rate, buffered_rate, args.expected_hz)
    comparison = {
        "expected_polling_rate_hz": args.expected_hz,
        "capture_seconds_per_method": args.seconds,
        "capture_order": list(order),
        "standard_event_count_rate_hz": standard_rate,
        "buffered_event_count_rate_hz": buffered_rate,
        "buffered_to_standard_ratio": round(
            buffered_rate / standard_rate if standard_rate > 0 else 0.0, 6
        ),
        "verdict": verdict,
        "interpretation": {
            "BOTH_BACKENDS_AGREE_BELOW_EXPECTED_RATE": (
                "The standard Python WM_INPUT reader is not the cause of the low rate."
            ),
            "STANDARD_BACKEND_MISSES_HIGH_FREQUENCY_EVENTS": (
                "The production standard reader misses events that the buffered reader recovers."
            ),
            "BOTH_BACKENDS_AGREE_WITH_EXPECTED_RATE": (
                "Both Python acquisition methods receive the expected rate."
            ),
            "BACKENDS_DISAGREE_REVIEW_RAW_OUTPUT": (
                "The difference is inconclusive; inspect movement continuity and raw files."
            ),
        }[verdict],
    }
    with (session_dir / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (session_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "motion_events",
                "capture_seconds",
                "motion_event_count_rate_hz",
            ),
        )
        writer.writeheader()
        for label in ("standard", "buffered"):
            summary = summaries[label]
            writer.writerow(
                {
                    "method": label,
                    "motion_events": summary["motion_events"],
                    "capture_seconds": summary["capture_seconds"],
                    "motion_event_count_rate_hz": summary[
                        "motion_event_count_rate_hz"
                    ],
                }
            )

    print("\nA/B RESULT")
    print(f"  Standard event-count rate: {standard_rate:.1f} Hz")
    print(f"  Buffered event-count rate: {buffered_rate:.1f} Hz")
    print(f"  Verdict: {verdict}")
    print(f"  Interpretation: {comparison['interpretation']}")
    print(f"  Artifacts: {session_dir}")

    if min(summaries[label]["motion_events"] for label in summaries) < 50:
        print(
            "Too few motion events in at least one capture; "
            "repeat while moving continuously."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

