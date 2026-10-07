"""Portable decoder regressions; owned rational tensors, no saved-result dependency."""
import copy
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import decoding
from exact import (cauchy_family, complete, from_orbits, kernel_graph, off_diagonal,
                   rational_orthogonal, synthesize, to_orbits)
from verify import _comm, verify


def definition_check(report, observed):
    """Dense full-coordinate reconstruction and enumerated disagreement lists."""
    n = len(observed)
    x = list(map(Q, report['completion']['particular']))
    directions = [list(map(Q, c['direction'])) for c in report['components']]
    supports = [[i for i in range(n) if z[i]] for z in directions]
    choices, selected, maxima = [], [], []
    for z, support in zip(directions, supports):
        ratios = [(observed[i] - x[i]) / z[i] for i in support]
        counts = Counter(ratios)
        largest = max(counts.values())
        selected.append(min(a for a in counts if counts[a] == largest))
        maxima.append(largest)
        finite = [(a, False) for a in sorted(counts)]
        choices.append(finite + [(max(counts) + 1, True)])
    estimate = [x[i] + sum((a * z[i] for a, z in zip(selected, directions)), Q(0))
                for i in range(n)]
    costs = []
    for choice in product(*choices):
        value = [x[i] + sum((a * z[i] for (a, _), z in zip(choice, directions)), Q(0))
                 for i in range(n)]
        costs.append((sum(a != b for a, b in zip(value, observed)),
                      any(unseen for _, unseen in choice)))
    minimum = min(cost for cost, _ in costs)
    finite_costs = [cost for cost, unseen in costs if not unseen]
    polynomial = [finite_costs.count(minimum + i) for i in range(sum(maxima) + 1)]
    sizes = ['infinite' if any(cost <= budget and unseen for cost, unseen in costs)
             else sum(cost <= budget for cost in finite_costs) for budget in range(n + 1)]
    return dict(estimate=list(map(str, estimate)), minimum_errors=minimum,
                minimum_list_size=finite_costs.count(minimum),
                finite_list_polynomial=polynomial,
                infinite_at_budget=next((i for i, value in enumerate(sizes)
                                         if value == 'infinite'), None),
                distance=min(map(len, supports), default=None), list_sizes=sizes)


def fixed_fibers():
    first = cauchy_family([Q(0), Q(1), Q(2)], [Q(1)] * 3, Q(1))
    left = cauchy_family([Q(0), Q(1)], [Q(1), Q(2)], Q(1))
    right = cauchy_family([Q(0), Q(2)], [Q(1), Q(1)], Q(1))
    second = from_orbits(5, [0] * 35)
    for i, j, k in product(range(2), repeat=3):
        second[i][j][k] = left[i][j][k]
        second[i+2][j+2][k+2] = right[i][j][k]
    second[4][4][4] = Q(3)
    return first, second


class SupportReconstructionTests(unittest.TestCase):
    def check_report(self, h, observed):
        saved = copy.deepcopy((h, observed))
        report = decoding.decode_diagonal(h, observed)
        self.assertTrue(verify({'n': len(h), 'orbits': to_orbits(h)}, report['completion']))
        if report['status'] == 'decoded':
            expected = definition_check(report, observed)
            for key, value in expected.items():
                if key != 'list_sizes':
                    self.assertEqual(report[key], value, key)
            self.assertEqual([decoding.list_size(report, s) for s in range(len(h)+1)],
                             expected['list_sizes'])
            tensor = complete(h, list(map(Q, report['estimate'])))
            self.assertTrue(all(_comm(tensor, (i, j, p, q)) == 0
                                for i, j in combinations(range(len(h)), 2)
                                for p, q in combinations(range(len(h)), 2)))
        self.assertEqual((h, observed), saved)
        return report

    def test_all_fixed_observations_and_budgets(self):
        for tensor in fixed_fibers():
            h = off_diagonal(tensor)
            x = list(map(Q, decoding.solve(h)['particular']))
            for word in product((-1, 0, 1), repeat=len(h)):
                self.check_report(h, [a + b for a, b in zip(x, word)])

    def test_singletons_rigid_and_inconsistent_fibers(self):
        for n in (*range(1, 9), 16):
            h = [[[Q(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
            observed = [Q((-1)**i * (i+1), 3) for i in range(n)]
            report = decoding.decode_diagonal(h, observed)
            self.assertEqual(report['estimate'], list(map(str, observed)))
            self.assertEqual(report['minimum_errors'], 0)
            self.assertEqual(report['minimum_list_size'], 1)
            self.assertEqual(report['infinite_at_budget'], 1)
            self.assertEqual(report['distance'], 1)
        rigid = synthesize(rational_orthogonal([1, 2, 3]), [Q(1), Q(2), Q(4)])
        observed = [rigid[i][i][i] + Q(i+1) for i in range(3)]
        self.assertEqual(self.check_report(off_diagonal(rigid), observed)['components'], [])
        inconsistent = from_orbits(3, [0, 1, 0, -1, 0, 0, 0, 1, -1, 0])
        self.assertEqual(self.check_report(inconsistent, [Q(0)]*3)['status'], 'inconsistent')
        with self.assertRaisesRegex(ValueError, 'diagonal dimension mismatch'):
            decoding.decode_diagonal(off_diagonal(rigid), [])

    def test_signed_rescaled_directions_and_modal_ties(self):
        tensor = cauchy_family([Q(-2), Q(0), Q(3), Q(7)],
                              [Q(2), Q(-3), Q(5), Q(7)], Q(1))
        h = off_diagonal(tensor)
        observed = [tensor[i][i][i] + z * a for i, (z, a) in
                    enumerate(zip((Q(2), Q(-3), Q(5), Q(7)), (Q(0), Q(1), Q(1), Q(0))))]
        graph = kernel_graph(h)
        for scale in (Q(1), Q(-3), Q(2, 7)):
            modified = copy.deepcopy(graph)
            modified['null_basis'] = [[str(scale * Q(v)) for v in z]
                                      for z in graph['null_basis']]
            with patch.object(decoding, 'kernel_graph', return_value=modified):
                report = self.check_report(h, observed)
            self.assertEqual(report['minimum_list_size'], 2)
            self.assertEqual(report['minimum_errors'], 2)


if __name__ == '__main__':
    unittest.main()
