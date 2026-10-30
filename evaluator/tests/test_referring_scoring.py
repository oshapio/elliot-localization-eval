"""Wrong-format predictions remain misses instead of aborting evaluation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import diagnostics, referring_accuracy


def row(targets, protocol="box"):
    return {"status": "ok", "benchmark_protocol": protocol,
            "gt": {"target": [[0, 0, 10, 10]]},
            "predictions": {"target": targets}}


class ReferringScoringTest(unittest.TestCase):
    def test_point_is_a_miss_without_removing_the_query(self):
        self.assertEqual(referring_accuracy([
            row([[5, 5]]), row([[0, 0, 10, 10]])]), 0.5)

    def test_valid_later_box_does_not_replace_invalid_first_prediction(self):
        self.assertEqual(referring_accuracy([row([[5, 5], [0, 0, 10, 10]])]), 0)

    def test_empty_prediction_is_still_a_miss(self):
        self.assertEqual(referring_accuracy([row([])]), 0)

    def test_dimension_diagnostics_distinguish_box_point_and_gui(self):
        self.assertEqual(diagnostics([row([[5, 5]])])["invalid_target_dimensions"], 1)
        self.assertEqual(diagnostics([row([[5, 5]], "point")])["invalid_target_dimensions"], 0)
        self.assertEqual(diagnostics([row([[5, 5]], "gui")])["invalid_target_dimensions"], 0)


if __name__ == "__main__":
    unittest.main()
