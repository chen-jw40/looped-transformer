import copy
import sys
import types
import unittest
from unittest.mock import patch

from scripts.evaluation_reporting import log_evaluation_tables


class FakeTable:
    def __init__(self, columns, data):
        self.columns = columns
        self.data = data


class FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


class FullWandbReportingTests(unittest.TestCase):
    def setUp(self):
        self.tasks = ["FOMC", "Py150", "MeetingBank", "20Minuten"]
        self.names = [f"{i:02d}_after_{task}" for i, task in enumerate(self.tasks, 1)]
        scores = [[0.8], [0.6, 70.0], [0.85, 65.0, 0.5], [0.75, 66.0, 0.45, 39.4]]
        metrics = ["accuracy", "similarity", "rouge-L", "sari"]
        self.summary = {
            "config": {"tasks": self.tasks, "eval_split": "test", "seed": 43},
            "performance_matrix": {
                name: {task: {"metric": metrics[j], "value": value}
                       for j, (task, value) in enumerate(zip(self.tasks, row))}
                for name, row in zip(self.names, scores)
            },
            "evaluations": {name: {task: {"samples": 100 + j}
                                   for j, task in enumerate(self.tasks[:i + 1])}
                            for i, name in enumerate(self.names)},
            "stage_status": {name: {"evaluation": "complete"} for name in self.names},
            "forgetting": {task: {} for task in self.tasks},
            "continual_aggregates": {
                "checkpoint_macro_average_available_tasks": dict(zip(self.names, [0.8, 0.65, 2 / 3, 0.5635])),
                "final_checkpoint": self.names[-1],
                "backward_transfer": -0.14 / 3,
                "backward_transfer_task_count": 3,
                "mean_final_forgetting_from_best": 0.0475,
            },
        }

    def report(self, summary=None):
        run = FakeRun()
        with patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Table=FakeTable)}):
            log_evaluation_tables(run, summary or self.summary)
        self.assertEqual(len(run.logged), 1)
        self.assertTrue(all(isinstance(value, FakeTable) for value in run.logged[0].values()))
        return run.logged[0]

    def cell(self, table, index, column):
        return table.data[index][table.columns.index(column)]

    def test_each_metric_has_its_own_matrices_and_only_tables_are_logged(self):
        original = copy.deepcopy(self.summary)
        payload = self.report()
        self.assertEqual(len(payload), 15)
        self.assertNotIn("eval/stage", payload)
        self.assertNotIn("eval/performance_matrix", payload)
        for metric, task in zip(["accuracy", "similarity", "rouge-L", "sari"], self.tasks):
            for kind in ["scores", "forgetting_from_acquisition", "forgetting_from_best"]:
                table = payload[f"eval/{kind}/{metric}"]
                self.assertEqual(len(table.data), 4)
                self.assertIn(task, table.columns)
                self.assertTrue(all(other not in table.columns for other in self.tasks if other != task))
        table = payload["eval/scores/accuracy"]
        self.assertEqual(self.cell(table, 0, "FOMC"), 80.0)
        self.assertEqual(self.cell(table, 3, "split"), "test")
        self.assertEqual(self.cell(table, 3, "seed"), 43)
        self.assertEqual(self.cell(table, 3, "status"), "complete")
        self.assertEqual(self.cell(table, 3, "FOMC/samples"), 100)
        self.assertEqual(self.cell(payload["eval/scores/similarity"], 1, "Py150"), 70.0)
        self.assertEqual(self.cell(payload["eval/scores/rouge-L"], 2, "MeetingBank"), 50.0)
        self.assertEqual(self.cell(payload["eval/scores/sari"], 3, "20Minuten"), 39.4)
        self.assertEqual(self.summary, original)

    def test_forgetting_uses_acquisition_and_best_after_acquisition(self):
        payload = self.report()
        acquisition = payload["eval/forgetting_from_acquisition/accuracy"]
        best = payload["eval/forgetting_from_best/accuracy"]
        self.assertAlmostEqual(self.cell(acquisition, 1, "FOMC"), 20.0)
        self.assertAlmostEqual(self.cell(acquisition, 2, "FOMC"), -5.0)
        self.assertAlmostEqual(self.cell(best, 2, "FOMC"), 0.0)
        self.assertAlmostEqual(self.cell(acquisition, 3, "FOMC"), 5.0)
        self.assertAlmostEqual(self.cell(best, 3, "FOMC"), 10.0)
        self.assertIsNone(self.cell(payload["eval/scores/similarity"], 0, "Py150"))
        self.assertIsNone(self.cell(payload["eval/forgetting_from_best/similarity"], 0, "Py150"))
        self.assertAlmostEqual(self.cell(payload["eval/forgetting_from_best/similarity"], 3, "Py150"), 4.0)
        self.assertAlmostEqual(self.cell(payload["eval/aggregates/macro_average_available_tasks"], 3, "value"), 56.35)
        self.assertAlmostEqual(self.cell(payload["eval/aggregates/backward_transfer"], 0, "value"), -14 / 3)
        self.assertAlmostEqual(self.cell(payload["eval/aggregates/mean_final_forgetting_from_best"], 0, "value"), 4.75)

    def test_final_only_results_leave_forgetting_and_bwt_unavailable(self):
        summary = copy.deepcopy(self.summary)
        final = self.names[-1]
        summary["performance_matrix"] = {final: summary["performance_matrix"][final]}
        summary["forgetting"] = {}
        summary["continual_aggregates"].update(backward_transfer=None,
                                               backward_transfer_task_count=0,
                                               mean_final_forgetting_from_best=None)
        payload = self.report(summary)
        for metric, task in zip(["accuracy", "similarity", "rouge-L", "sari"], self.tasks):
            self.assertEqual(self.cell(payload[f"eval/scores/{metric}"], 0, "stage"), 4)
            for kind in ["forgetting_from_acquisition", "forgetting_from_best"]:
                self.assertIsNone(self.cell(payload[f"eval/{kind}/{metric}"], 0, task))
        self.assertIsNone(self.cell(payload["eval/aggregates/backward_transfer"], 0, "value"))
        self.assertIsNone(self.cell(payload["eval/aggregates/mean_final_forgetting_from_best"], 0, "value"))

    def test_missing_acquisition_is_not_replaced_with_first_available_score(self):
        summary = copy.deepcopy(self.summary)
        del summary["performance_matrix"][self.names[1]]
        payload = self.report(summary)
        self.assertIsNone(self.cell(payload["eval/forgetting_from_best/similarity"], 2, "Py150"))

    def test_disabled_reporting_does_not_import_wandb(self):
        with patch.dict(sys.modules, {"wandb": None}):
            log_evaluation_tables(None, {})


if __name__ == "__main__":
    unittest.main()
