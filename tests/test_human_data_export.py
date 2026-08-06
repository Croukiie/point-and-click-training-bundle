import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.experiment.export_human_amortizer_data import (
    AMORTIZER_FEATURES,
    export_sessions,
)
from src.utils.trajectory_features import CURVATURE_FEATURE_NAMES


class HumanDataExportTests(unittest.TestCase):
    def test_export_produces_aligned_stats_and_20hz_trajectory(self):
        with tempfile.TemporaryDirectory(prefix="pnc_human_export_") as temporary:
            root = Path(temporary) / "human"
            session_dir = root / "session_1"
            session_dir.mkdir(parents=True)
            (session_dir / "session.json").write_text(
                json.dumps({
                    "session_id": "session_1",
                    "participant": {"id": "P001"},
                    "task": {
                        "config": {
                            "experiment": {
                                "protocol_version": "target_radius_matched_v2"
                            }
                        }
                    },
                    "hardware": {
                        "declared_mouse_polling_hz": 1000,
                        "cpi_setting_method": "mouse_software",
                        "mousepad_condition": "firm",
                    },
                }),
                encoding="utf-8",
            )

            trial = {
                "session_id": "session_1",
                "trial_id": "7",
                "phase": "measured",
                "trial_stage": "task",
                "block_index": "1",
                "cpi": "800",
                "cpi_confirmed": "True",
                "valid": "True",
                "outcome": "success",
                "success": "True",
                "reaction_time_s": "0.1",
                "movement_time_s": "0.1",
                "cursor_start_x_m": "0.0",
                "cursor_start_y_m": "0.0",
                "target_x_m": "0.1",
                "target_y_m": "0.0",
                "target_radius_m": "0.01",
                "click_x_m": "0.099",
                "click_y_m": "0.0",
                "click_error_m": "0.001",
                "path_length_m": "0.101",
                "mousepad_condition": "firm",
                "input_backend": "windows_raw_input",
                **{name: "0.02" for name in CURVATURE_FEATURE_NAMES},
            }
            with (session_dir / "trials.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trial))
                writer.writeheader()
                writer.writerow(trial)

            raw_rows = [
                {"trial_id": "7", "time_from_onset_ns": "100000000", "cursor_x_m": "0.001", "cursor_y_m": "0", "speed_m_s": "0.2"},
                {"trial_id": "7", "time_from_onset_ns": "150000000", "cursor_x_m": "0.05", "cursor_y_m": "0.002", "speed_m_s": "0.5"},
                {"trial_id": "7", "time_from_onset_ns": "200000000", "cursor_x_m": "0.099", "cursor_y_m": "0", "speed_m_s": "0.3"},
            ]
            with (session_dir / "raw_mouse_events.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
                writer.writeheader()
                writer.writerows(raw_rows)

            output = Path(temporary) / "export"
            manifest = export_sessions(
                root,
                output,
                trajectory_hz=20,
                protocol_version="target_radius_matched_v2",
            )

            self.assertEqual(manifest["exported_trial_count"], 1)
            self.assertEqual(manifest["trials_per_cpi"], {"800": 1})
            self.assertEqual(
                manifest["protocol_version_filter"],
                "target_radius_matched_v2",
            )
            self.assertEqual(manifest["source_sessions"], ["session_1"])
            with (output / "human_amortizer_stats.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                stats = list(csv.DictReader(handle))
            self.assertEqual(len(stats), 1)
            self.assertTrue(set(AMORTIZER_FEATURES).issubset(stats[0]))
            self.assertEqual(float(stats[0]["max_cursor_speed"]), 0.5)

            with (output / "human_amortizer_trajectories.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                trajectory = list(csv.DictReader(handle))
            self.assertEqual(len(trajectory), 3)
            self.assertEqual(float(trajectory[0]["timestamp"]), 0.0)
            self.assertAlmostEqual(float(trajectory[1]["timediff"]), 0.05)
            self.assertEqual({row["trial_uid"] for row in trajectory}, {"session_1:7"})

    def test_unconfirmed_cpi_is_excluded_by_default(self):
        trial = {
            "phase": "measured",
            "trial_stage": "task",
            "valid": "True",
            "outcome": "success",
            "cpi_confirmed": "False",
        }
        from src.experiment.export_human_amortizer_data import _trial_selection_reason

        self.assertEqual(
            _trial_selection_reason(
                trial,
                include_practice=False,
                allow_unconfirmed_cpi=False,
            ),
            "cpi_not_confirmed",
        )


if __name__ == "__main__":
    unittest.main()
