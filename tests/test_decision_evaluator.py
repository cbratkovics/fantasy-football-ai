import unittest

from backend.evaluation.decision_evaluator import evaluate_rows


class DecisionEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"player_id": "a", "season": 2024, "week": 1, "position": "QB", "prediction": 12, "actual": 10, "decision_score": .8, "prediction_floor": 8},
            {"player_id": "b", "season": 2024, "week": 1, "position": "QB", "prediction": 20, "actual": 30, "decision_score": .4, "prediction_floor": 14},
            {"player_id": "a", "season": 2024, "week": 2, "position": "QB", "prediction": 14, "actual": 12, "decision_score": .9, "prediction_floor": 11},
            {"player_id": "b", "season": 2024, "week": 2, "position": "QB", "prediction": 25, "actual": 20, "decision_score": .6, "prediction_floor": 18},
        ]

    def test_report_compares_model_and_causal_baseline(self) -> None:
        report = evaluate_rows(self.rows, (2024, 2), thresholds=(.5, .7))
        self.assertEqual(report["model"], {"n": 2, "mae": 3.5, "median_absolute_error": 3.5})
        # Week-two baselines are week-one player outcomes (10 and 30), never week-two outcomes.
        self.assertEqual(report["baseline"]["mae"], 6)
        self.assertEqual(report["policy_sweep"][1]["recommended"], 1)
        self.assertEqual(report["policy_sweep"][1]["downside_rate"], 0)

    def test_rejects_missing_contract_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            evaluate_rows([{"player_id": "a"}], (2024, 1))

    def test_rejects_empty_test_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty test set"):
            evaluate_rows(self.rows, (2025, 1))


if __name__ == "__main__":
    unittest.main()
