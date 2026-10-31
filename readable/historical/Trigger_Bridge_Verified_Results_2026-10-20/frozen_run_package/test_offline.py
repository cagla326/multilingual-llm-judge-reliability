"""Pure software tests: no fixture sentences/labels enter the experiment."""

import unittest
from common import validate, capped, trigger, rows, ROOT
from analyze_bridge import check_prediction, measures, intervals


class Tests(unittest.TestCase):
    def test_original_pipeline_and_inputs(self):
        _, meta, claims = validate()
        self.assertEqual(len(meta), 900)
        self.assertEqual(len(claims), 2489)
        self.assertEqual(sum(not r['claim_keys'] for r in meta), 3)
        self.assertTrue(all('human' not in r and 'baseline' not in r for r in claims))

    def test_trigger_boundaries_and_symmetry(self):
        for b, c, want in [
            (75, 50, 62.5),
            (50, 75, 62.5),
            (100, 0, 50),
            (0, 100, 50),
            (74, 0, 74),
            (0, 74, 0),
            (75, 51, 75),
            (51, 75, 51),
            (50, 50, 50),
        ]:
            self.assertEqual(trigger(b, c), want)

    def test_inherited_cap_priority(self):
        self.assertEqual(capped([]), 0)
        self.assertEqual(capped(['entailment'] * 9 + ['contradiction']), 25)
        self.assertEqual(capped(['entailment'] * 9 + ['neutral']), 50)
        self.assertEqual(capped(['neutral']), 0)
        self.assertEqual(capped(['entailment']), 100)
        with self.assertRaises(ValueError):
            capped(['partially_supported'])

    def test_reject_bad_prediction(self):
        with self.assertRaises(AssertionError):
            check_prediction(
                {
                    'label': 'neutral',
                    'restricted_logprobs': {
                        'entailment': 0.0,
                        'contradiction': 0.0,
                        'neutral': 0.0,
                    },
                }
            )

    def test_metric_reproduction(self):
        _, meta, _ = validate()
        y = [r['human'] for r in meta]
        b = [r['baseline'] for r in meta]
        f = [r['original_final'] for r in meta]
        self.assertAlmostEqual(measures(y, b)['spearman'], 0.25215132897979803, 12)
        self.assertAlmostEqual(measures(y, b)['mae'], 34.60763333333334, 10)
        self.assertGreater(measures(y, f)['spearman'], measures(y, b)['spearman'])

    def test_identical_condition_cluster_interval_zero(self):
        _, meta, _ = validate()
        v = [r['original_final'] for r in meta]
        ci = intervals(
            meta,
            {'trained3_selective': v, 'original4_selective': v, 'base3_selective': v},
            20,
        )
        for r in ci.values():
            self.assertEqual(r['spearman_delta_95_percentile_ci'], [0, 0])
            self.assertEqual(r['mae_delta_95_percentile_ci'], [0, 0])


if __name__ == '__main__':
    unittest.main()
