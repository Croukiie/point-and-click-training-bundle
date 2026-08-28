"""Measure the effective Windows Raw Input mouse polling rate.

This diagnostic is intentionally independent from the web experiment and from
the 20 Hz trajectory preprocessing used by the amortizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .raw_input_win32 import RawInputMouse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARD_POLLING_RATES_HZ = (125, 250, 500, 1000, 2000, 4000, 8000)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _summarize_events(events, capture_seconds: float, expected_hz: float) -> dict:
    motion_events = [event for event in events if event.dx != 0 or event.dy != 0]
    intervals_ms = [
        (current.timestamp_ns - previous.timestamp_ns) / 1_000_000.0
        for previous, current in zip(motion_events, motion_events[1:])
        if current.timestamp_ns > previous.timestamp_ns
    ]
    # Long pauses indicate that the mouse was briefly stationary. They are not
    # device polling intervals and must not lower the polling estimate.
    active_intervals_ms = [interval for interval in intervals_ms if interval <= 50.0]

    median_ms = statistics.median(active_intervals_ms) if active_intervals_ms else None
    median_rate_hz = 1000.0 / median_ms if median_ms and median_ms > 0 else None
    span_rate_hz = None
    if len(motion_events) >= 2:
        span_seconds = (
            motion_events[-1].timestamp_ns - motion_events[0].timestamp_ns
        ) / 1_000_000_000.0
        if span_seconds > 0:
            span_rate_hz = (len(motion_events) - 1) / span_seconds

    rounded_intervals = Counter(round(interval, 1) for interval in active_intervals_ms)
    mode_ms = rounded_intervals.most_common(1)[0][0] if rounded_intervals else None
    nearest_standard_hz = None
    if median_rate_hz is not None:
        nearest_standard_hz = min(
            STANDARD_POLLING_RATES_HZ,
            key=lambda rate: abs(rate - median_rate_hz),
        )

    device_counts = Counter(
        (event.device_handle, event.device_name) for event in motion_events
    )
    devices = [
        {
            "device_handle": handle,
            "device_name": name,
            "motion_events": count,
        }
        for (handle, name), count in device_counts.most_common()
    ]

    compatible = (
        median_rate_hz is not None and median_rate_hz >= expected_hz * 0.75
    )
    return {
        "capture_seconds": round(capture_seconds, 6),
        "expected_polling_rate_hz": expected_hz,
        "total_raw_input_events": len(events),
        "motion_events": len(motion_events),
        "active_intervals": len(active_intervals_ms),
        "effective_rate_from_event_span_hz": _round_or_none(span_rate_hz, 3),
        "effective_rate_from_median_interval_hz": _round_or_none(median_rate_hz, 3),
        "nearest_standard_polling_rate_hz": nearest_standard_hz,
        "interval_median_ms": _round_or_none(median_ms),
        "interval_mode_rounded_0_1_ms": mode_ms,
        "interval_p05_ms": _round_or_none(_percentile(active_intervals_ms, 0.05)),
        "interval_p95_ms": _round_or_none(_percentile(active_intervals_ms, 0.95)),
        "fraction_intervals_0_5_to_1_5_ms": round(
            sum(0.5 <= value <= 1.5 for value in active_intervals_ms)
            / max(1, len(active_intervals_ms)),
            6,
        ),
        "fraction_intervals_6_to_10_ms": round(
            sum(6.0 <= value <= 10.0 for value in active_intervals_ms)
            / max(1, len(active_intervals_ms)),
            6,
        ),
        "compatible_with_expected_rate": compatible,
        "devices": devices,
    }


def _write_events_csv(path: Path, events) -> None:
    fields = (
        "timestamp_ns",
        "device_handle",
        "device_name",
        "raw_flags",
        "button_flags",
        "button_data",
        "dx",
        "dy",
        "os_cursor_x_px",
        "os_cursor_y_px",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: getattr(event, field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the effective mouse polling rate received by Windows Raw Input."
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--expected-hz", type=float, default=125.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "diagnostics" / "mouse_polling",
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.expected_hz <= 0:
        parser.error("--expected-hz must be positive")

    print("Windows Raw Input polling diagnostic")
    print("Move the mouse continuously in large circles during the capture.")
    print("Do not click. The capture starts automatically after the countdown.")

    mouse = RawInputMouse()
    mouse.start()
    events = []
    try:
        for remaining in range(3, 0, -1):
            print(f"Starting in {remaining}...", flush=True)
            time.sleep(1.0)
        mouse.drain()
        print(f"CAPTURING for {args.seconds:g} seconds...", flush=True)
        start = time.perf_counter()
        deadline = start + args.seconds
        while time.perf_counter() < deadline:
            events.extend(mouse.drain())
            time.sleep(0.001)
        events.extend(mouse.drain())
        elapsed = time.perf_counter() - start
    finally:
        mouse.stop()

    summary = _summarize_events(events, elapsed, args.expected_hz)
    session_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    csv_path = session_dir / "raw_input_events.csv"
    json_path = session_dir / "polling_summary.json"
    _write_events_csv(csv_path, events)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("\nResult")
    print(f"  Raw events: {summary['total_raw_input_events']}")
    print(f"  Motion events: {summary['motion_events']}")
    rate = summary["effective_rate_from_median_interval_hz"]
    median_ms = summary["interval_median_ms"]
    if rate is None:
        print("  Effective polling rate: unavailable (not enough movement)")
    else:
        print(f"  Median interval: {median_ms:.3f} ms")
        print(f"  Estimated effective polling rate: {rate:.1f} Hz")
        print(
            "  Nearest standard rate: "
            f"{summary['nearest_standard_polling_rate_hz']} Hz"
        )
        if summary["compatible_with_expected_rate"]:
            print(f"  Verdict: compatible with an expected {args.expected_hz:g} Hz setting")
        else:
            print(f"  Verdict: clearly below the expected {args.expected_hz:g} Hz setting")
    print(f"  Summary: {json_path}")
    print(f"  Raw data: {csv_path}")

    if summary["motion_events"] < 50:
        print("\nToo few motion events. Repeat while moving continuously.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
