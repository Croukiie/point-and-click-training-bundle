"""Measure physical mouse CPI from Windows Raw Input counts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
from statistics import mean, stdev

from .raw_input_win32 import RawInputMouse


ROOT = Path(__file__).resolve().parents[2]


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_settings(value: str) -> list[int]:
    settings = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not settings or any(item <= 0 for item in settings):
        raise argparse.ArgumentTypeError(
            "settings must be comma-separated positive integers"
        )
    return settings


def default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "data/diagnostics" / f"physical_cpi_calibration_{timestamp}.csv"


def run(distance_cm: float, settings: list[int], repeats: int, output: Path) -> None:
    distance_inches = distance_cm / 2.54
    rows: list[dict[str, object]] = []
    mouse = RawInputMouse()
    mouse.start()

    print("Windows Raw Input CPI calibration")
    print(f"Physical movement distance: {distance_cm:g} cm")
    print("Use only the mouse being tested. Do not touch another mouse or touchpad.")

    try:
        for nominal_cpi in settings:
            input(
                f"\nSet the mouse to {nominal_cpi} CPI. Place the sensor on "
                "the first mark, then press Enter."
            )
            estimates: list[float] = []

            for repetition in range(1, repeats + 1):
                # Discard movement used to return the mouse to the first mark.
                mouse.drain()
                input(
                    f"Trial {repetition}/{repeats}: move exactly {distance_cm:g} cm "
                    "without lifting the mouse, then press Enter."
                )
                events = mouse.drain()
                dx_counts = sum(event.dx for event in events)
                dy_counts = sum(event.dy for event in events)
                net_counts = math.hypot(dx_counts, dy_counts)
                measured_cpi = net_counts / distance_inches
                error_pct = 100.0 * (measured_cpi / nominal_cpi - 1.0)
                device_names = sorted(
                    {event.device_name for event in events if event.device_name}
                )
                estimates.append(measured_cpi)
                rows.append(
                    {
                        "nominal_cpi": nominal_cpi,
                        "repetition": repetition,
                        "distance_cm": distance_cm,
                        "dx_counts": dx_counts,
                        "dy_counts": dy_counts,
                        "net_counts": net_counts,
                        "measured_cpi": measured_cpi,
                        "error_pct": error_pct,
                        "raw_event_count": len(events),
                        "device_names": ";".join(device_names),
                    }
                )
                print(
                    f"  dx={dx_counts}, dy={dy_counts}, net={net_counts:.0f}, "
                    f"measured CPI={measured_cpi:.1f}, error={error_pct:+.1f}%"
                )
                if len(device_names) > 1:
                    print(
                        "  WARNING: events from multiple pointing devices were "
                        "detected. Repeat without touching the other device."
                    )
                input("Return the sensor to the first mark, then press Enter.")

            average = mean(estimates)
            variability = stdev(estimates) if len(estimates) > 1 else 0.0
            print(
                f"RESULT {nominal_cpi} CPI: mean={average:.1f}, "
                f"SD={variability:.1f}, "
                f"mean error={100.0 * (average / nominal_cpi - 1.0):+.1f}%"
            )
    finally:
        mouse.stop()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results: {output.resolve()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-cm", type=positive_float, default=20.0)
    parser.add_argument("--settings", type=parse_settings, default=[400, 800, 1600])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run(
        distance_cm=args.distance_cm,
        settings=args.settings,
        repeats=args.repeats,
        output=args.output or default_output(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
