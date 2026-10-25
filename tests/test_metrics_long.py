import unittest

from ouro_pipeline.metrics import evaluate_task


class LongMetricTests(unittest.TestCase):
    def test_long_rouge_does_not_hit_python_recursion_limit(self):
        words = [f"word{index}" for index in range(1200)]
        text = " ".join(words)
        result = evaluate_task("MeetingBank", [text], [text])
        self.assertGreater(result["rouge-L"], 0.99)

    def test_sentence_less_rouge_input_is_zero_not_an_exception(self):
        result = evaluate_task("MeetingBank", ["..."], ["..."])
        self.assertEqual(result["rouge-L"], 0.0)

    def test_whitespace_prediction_is_a_format_failure(self):
        result = evaluate_task("MeetingBank", ["   "], ["reference"])
        self.assertEqual(result["format_failures"], 1)
        self.assertEqual(result["rouge-L"], 0.0)


if __name__ == "__main__":
    unittest.main()
