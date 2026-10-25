import unittest

from ouro_pipeline.sari import corpus_sari, sentence_sari


class SariTests(unittest.TestCase):
    def test_huggingface_reference_example(self):
        score = sentence_sari(
            "About 95 species are currently accepted.",
            "About 95 you now get in.",
            [
                "About 95 species are currently known.",
                "About 95 species are now accepted.",
                "95 species are now accepted.",
            ],
        )
        self.assertAlmostEqual(score, 26.953601953601954, places=10)

    def test_perfect_match(self):
        sentence = "About 95 species are currently accepted."
        self.assertEqual(sentence_sari(sentence, sentence, [sentence]), 100.0)

    def test_corpus_validation(self):
        with self.assertRaises(ValueError):
            corpus_sari(["source"], [], ["reference"])


if __name__ == "__main__":
    unittest.main()
