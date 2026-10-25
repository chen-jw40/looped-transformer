import unittest

from ouro_pipeline.training import continual_aggregates, continual_metrics


class ContinualMetricsTests(unittest.TestCase):
    def test_forgetting_bwt_and_scale_normalization(self):
        evaluations = {
            "00_initial": {"FOMC": {"accuracy": 0.1}, "Py150": {"similarity": 5.0}},
            "01_after_FOMC": {"FOMC": {"accuracy": 0.8}},
            "02_after_Py150": {"FOMC": {"accuracy": 0.5}, "Py150": {"similarity": 60.0}},
        }
        matrix, forgetting = continual_metrics(evaluations, ["FOMC", "Py150"])
        aggregate = continual_aggregates(matrix, forgetting, ["FOMC", "Py150"])
        self.assertAlmostEqual(forgetting["FOMC"]["final_forgetting_from_best"], 0.3)
        self.assertAlmostEqual(aggregate["backward_transfer"], -0.3)
        self.assertAlmostEqual(aggregate["final_macro_average"], 0.55)
        self.assertAlmostEqual(aggregate["mean_final_forgetting_from_best"], 0.15)


if __name__ == "__main__":
    unittest.main()
