"""Plot-only lineage checks; no solver, experiment, or plotting dependency."""
from pathlib import Path
from fractions import Fraction as Q
import ast
import csv
import itertools
import json
import math
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PlotSources(unittest.TestCase):
    def test_saved_certificates_bind_all_raw_errors_and_radii(self):
        with (ROOT / 'results/noise.csv').open(newline='') as stream:
            rows = {r['case']: r for r in csv.DictReader(stream)}
        saved = json.loads((ROOT / 'results/noise-certificates.json').read_text())
        self.assertEqual(len(saved), 36)
        self.assertEqual({r['case'] for r in saved}, set(rows))
        for record in saved:
            raw, cert = rows[record['case']], record['certificate']
            n = int(raw['n'])
            orbits = list(itertools.combinations_with_replacement(range(n), 3))
            truth = {key: Q(value) for key, value in zip(orbits, record['truth'])}
            error = sum((Q(cert['candidate_diagonal'][i]) - truth[i, i, i])**2 for i in range(n))
            self.assertEqual(error, Q(record['actual_error_squared']))
            self.assertEqual(error, Q(raw['error_squared']))
            self.assertEqual(Q(cert['diagonal_radius']), Q(raw['diagonal_radius']))
            self.assertEqual(cert['status'], 'conditional_bound')

    def test_all_amplitudes_share_plotted_quantities(self):
        with (ROOT / 'results/noise.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)
        groups = {}
        for row in rows:
            groups.setdefault((int(row['n']), int(row['noise_exponent'])), []).append(row)
        self.assertEqual(set(groups), {(n, p) for n in (3, 5, 7) for p in (10, 14, 18)})
        for group in groups.values():
            self.assertEqual({int(r['amplitude']) for r in group}, {1, 2**10, 2**20, 2**40})
            for key in ('selected_support', 'error_squared', 'diagonal_radius'):
                self.assertEqual(len({r[key] for r in group}), 1)
            for row in group:
                self.assertLessEqual(Q(row['error_squared']), Q(row['diagonal_radius'])**2)

    def test_processed_points_are_exact_export_conversions(self):
        with (ROOT / 'results/noise.csv').open(newline='') as stream:
            raw = [r for r in csv.DictReader(stream) if r['amplitude'] == '1']
        for n in (3, 5, 7):
            lines = (ROOT / f'results/plot-data/noise-{n}.dat').read_text().splitlines()
            self.assertEqual(lines[0], 'scale error radius')
            expected = []
            for row in raw:
                if int(row['n']) == n:
                    values = (2.**-int(row['noise_exponent']), math.sqrt(float(Q(row['error_squared']))), float(Q(row['diagonal_radius'])))
                    expected.append(' '.join(format(v, '.12g') for v in values))
            self.assertEqual(lines[1:], expected)

    def test_optional_route_keeps_current_snapshot(self):
        tree = ast.parse((ROOT / 'scripts/plot_retained.py').read_text(encoding='utf-8'))
        snapshots = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SNAPSHOTS' for t in n.targets))
        for path, snapshot in ast.literal_eval(snapshots).items():
            self.assertEqual((ROOT / 'results/plot-data' / Path(path).name).read_text(), snapshot)
        source = (ROOT / 'scripts/plot_retained.py').read_text()
        for option in ('--input-dir', '--output-dir', '--font-afm-dir', '--font-pfb-dir'):
            self.assertIn(option, source)

if __name__ == '__main__':
    unittest.main()
