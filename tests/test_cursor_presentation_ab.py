import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.experiment.compare_cursor_presentation_ab import (
    FEATURES,
    feature_statistics,
    load_session,
    mode_differences,
)


class CursorPresentationABTests(unittest.TestCase):
    def _session(self, root: Path, mode: str, offset: float) -> Path:
        session = root / mode
        session.mkdir()
        (session / "session.json").write_text(
            json.dumps(
                {
                    "input_backend": "windows_raw_input",
                    "hardware": {"cursor_presentation_mode": mode},
                }
            ),
            encoding="utf-8",
        )
        rows = []
        for cpi in (400, 800, 1600):
            for repeat in range(2):
                row = {
                    "phase": "measured",
                    "trial_stage": "task",
                    "valid": True,
                    "cpi": cpi,
                    "native_alignment_p95_m": 0.001 if mode == "native_windows" else None,
                }
                row.update(
                    {
                        feature: (
                            float((cpi / 400) + repeat + offset)
                            if feature != "success"
                            else float(repeat == 0)
                        )
                        for feature in FEATURES
                    }
                )
                rows.append(row)
        pd.DataFrame(rows).to_csv(session / "trials.csv", index=False)
        return session

    def test_sessions_are_labeled_and_statistics_compare_modes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            virtual_path = self._session(root, "virtual_raw_input", 0.0)
            native_path = self._session(root, "native_windows", 1.0)
            virtual, _ = load_session(virtual_path, "virtual_raw_input")
            native, _ = load_session(native_path, "native_windows")
            statistics = feature_statistics(pd.concat([virtual, native], ignore_index=True))
            differences = mode_differences(statistics)
            self.assertEqual(len(virtual), 6)
            self.assertEqual(len(native), 6)
            self.assertEqual(len(statistics), len(FEATURES) * 3 * 2)
            movement = differences[
                differences["feature"].eq("movement_time_s")
            ]
            self.assertTrue((movement["native_minus_virtual"] > 0).all())


if __name__ == "__main__":
    unittest.main()
