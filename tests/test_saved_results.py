"""Regressions for saved-result checking rather than producer generation."""
from __future__ import annotations

import copy
import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import check


class SavedResultRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structured = json.loads((ROOT / 'results' / 'structured.json').read_text())
        cls.structured_summary = json.loads((ROOT / 'results' / 'structured-summary.json').read_text())
        cls.noise = json.loads((ROOT / 'results' / 'noise-certificates.json').read_text())
        with (ROOT / 'results' / 'noise.csv').open(newline='') as handle:
            cls.noise_rows = list(csv.DictReader(handle))
        cls.noise_summary = json.loads((ROOT / 'results' / 'noise-summary.json').read_text())
        cls.decoding = json.loads((ROOT / 'results' / 'decoding.json').read_text())
        cls.decoding_summary = json.loads((ROOT / 'results' / 'decoding-summary.json').read_text())
        cls.mutations = json.loads((ROOT / 'results' / 'mutations.json').read_text())
        cls.mutation_summary = json.loads((ROOT / 'results' / 'mutations-summary.json').read_text())
        cls.negative = json.loads((ROOT / 'results' / 'negative-controls.json').read_text())

    @classmethod
    def tearDownClass(cls):
        import gc
        for name in ('structured', 'structured_summary', 'noise', 'noise_rows', 'noise_summary', 'decoding', 'decoding_summary',
                     'mutations', 'mutation_summary', 'negative'):
            if hasattr(cls, name):
                delattr(cls, name)
        gc.collect()

    def test_structured_truth_binding_and_component_distance(self):
        design = check.structured_design()
        check.check_structured_records(
            self.structured,
            self.structured_summary,
            design=design,
        )

        forged = copy.deepcopy(self.structured)
        index = next(i for i, record in enumerate(forged) if record['case'] == 'distance-2-2')
        forged[index] = {
            'case': 'distance-2-2',
            'input': {'n': 2, 'orbits': ['0', '0', '0', '0']},
            'truth': ['1', '0', '0', '1'],
            'certificate': {
                'status': 'ambiguous',
                'particular': ['0', '0'],
                'null_basis': [['1', '1'], ['1', '-1']],
                'rank': 0,
                'pivot_equations': [],
                'pivot_columns': [],
            },
            'tensor_rank': 2,
            'distance': 2,
        }
        free, rank = check.independent_free_components(check.decode(forged[index]['input']))
        self.assertEqual(rank, 0)
        self.assertEqual([component['support'] for component in free], [(0,), (1,)])
        self.assertEqual(min(len(component['support']) for component in free), 1)
        with self.assertRaisesRegex(ValueError, 'structured (truth|input)/design mismatch|component distance'):
            check.check_structured_records(forged, self.structured_summary, design=design)

        mismatched = copy.deepcopy(self.structured)
        record = next(record for record in mismatched if record['case'] == 'distance-2-1')
        record['truth'] = ['1', '0', '0', '1']
        with self.assertRaisesRegex(ValueError, 'structured truth/design mismatch'):
            check.check_structured_records(mismatched, self.structured_summary, design=design)

        duplicate = copy.deepcopy(self.structured)
        duplicate[-1]['case'] = duplicate[-2]['case']
        with self.assertRaisesRegex(ValueError, 'duplicate structured case'):
            check.check_structured_records(duplicate, self.structured_summary, design=design)

    def test_noise_key_sets_reject_duplicate_omission_and_mislabel(self):
        check.check_noise_records(self.noise, self.noise_rows, self.noise_summary)

        duplicate_certificates = copy.deepcopy(self.noise)
        duplicate_certificates[-1] = copy.deepcopy(duplicate_certificates[0])
        with self.assertRaisesRegex(ValueError, 'duplicate noise certificate case'):
            check.check_noise_records(duplicate_certificates, self.noise_rows, self.noise_summary)

        duplicate_csv = copy.deepcopy(self.noise_rows)
        duplicate_csv[-1] = copy.deepcopy(duplicate_csv[0])
        with self.assertRaisesRegex(ValueError, 'duplicate noise csv case'):
            check.check_noise_records(self.noise, duplicate_csv, self.noise_summary)

        with self.assertRaisesRegex(ValueError, 'noise case count'):
            check.check_noise_records(self.noise[:-1], self.noise_rows, self.noise_summary)

        with self.assertRaisesRegex(ValueError, 'noise csv count'):
            check.check_noise_records(self.noise, self.noise_rows[:-1], self.noise_summary)

        mislabeled = copy.deepcopy(self.noise_rows)
        target = next(row for row in mislabeled if row['case'] == 'noise-3-1-10-1')
        target['noise_exponent'] = '11'
        with self.assertRaisesRegex(ValueError, 'noise CSV n/s/p/amplitude mismatch'):
            check.check_noise_records(self.noise, mislabeled, self.noise_summary)

    def test_decoding_saved_records_are_replayed_per_budget(self):
        check.check_decoding_records(self.decoding, self.decoding_summary)

        wrong_budget = copy.deepcopy(self.decoding)
        target = next(record for record in wrong_budget['cases'] if record['case'] == 'word-0-1')
        self.assertEqual(target['list_sizes'], [0, 1, 2, 'infinite'])
        target['list_sizes'][2] = 999
        with self.assertRaisesRegex(ValueError, 'decoding budget list mismatch'):
            check.check_decoding_records(wrong_budget, self.decoding_summary)

        wrong_observation = copy.deepcopy(self.decoding)
        target = next(record for record in wrong_observation['cases'] if record['case'] == 'word-1-0')
        target['observed_diagonal'][0] = '999'
        with self.assertRaisesRegex(ValueError, 'decoding observation mismatch'):
            check.check_decoding_records(wrong_observation, self.decoding_summary)

        wrong_labels = copy.deepcopy(self.decoding)
        first = next(record for record in wrong_labels['cases'] if record['case'] == 'word-0-0')
        second = next(record for record in wrong_labels['cases'] if record['case'] == 'word-0-1')
        first['case'], second['case'] = second['case'], first['case']
        with self.assertRaisesRegex(ValueError, 'decoding (observation|budget list) mismatch'):
            check.check_decoding_records(wrong_labels, self.decoding_summary)

    def test_mutation_case_coverage_flags_and_summary(self):
        posterior = next(record['certificate'] for record in self.noise
                         if record['case'] == 'noise-3-1-10-1')
        check.check_mutation_records(self.mutations, posterior, self.mutation_summary)
        duplicate = copy.deepcopy(self.mutations)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, 'mutation case set mismatch'):
            check.check_mutation_records(duplicate, posterior, self.mutation_summary)
        unknown = copy.deepcopy(self.mutations)
        unknown[-1]['case'] = 'unperformed-control'
        with self.assertRaisesRegex(ValueError, 'mutation case set mismatch'):
            check.check_mutation_records(unknown, posterior, self.mutation_summary)
        truthy = copy.deepcopy(self.mutations)
        truthy[-1]['rejected'] = 'False'
        with self.assertRaisesRegex(ValueError, 'mutation rejection flags'):
            check.check_mutation_records(truthy, posterior, self.mutation_summary)
        summary = dict(self.mutation_summary, rejected=9)
        with self.assertRaisesRegex(ValueError, 'mutation summary mismatch'):
            check.check_mutation_records(self.mutations, posterior, summary)

    def test_negative_controls_require_refusal_status_and_premises(self):
        check.check_negative_controls(self.negative)
        success = copy.deepcopy(self.negative)
        success['distance_two_budget_one']['status'] = 'conditional_bound'
        with self.assertRaisesRegex(ValueError, 'negative-control status/reason mismatch'):
            check.check_negative_controls(success)
        wrong_reason = copy.deepcopy(self.negative)
        wrong_reason['excessive_noise_bound']['reason'] = 'singular_restricted_system'
        with self.assertRaisesRegex(ValueError, 'negative-control status/reason mismatch'):
            check.check_negative_controls(wrong_reason)
        missing = copy.deepcopy(self.negative)
        missing.pop('excessive_noise_bound')
        with self.assertRaisesRegex(ValueError, 'negative-control status/reason mismatch'):
            check.check_negative_controls(missing)


if __name__ == '__main__':
    unittest.main()
