"""Deterministic adversarial regressions for proof-critical identities."""
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from exact import (
    cauchy_family,
    distance_family,
    equations,
    kernel_graph,
    off_diagonal,
    solve,
    to_orbits,
)
from stability import certify
from verify import _rank, _row, decode, verify, verify_stability


class AdversarialRegressions(unittest.TestCase):
    def test_optimized_python_is_rejected(self):
        """The campaign must not silently drop assertion-based invariants."""
        run_py = Path(__file__).resolve().parents[1] / 'run.py'
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, '-O', str(run_py), '--suite', 'mutations', '--out', tmp],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('optimized Python', proc.stdout)

    def test_equation_builder_matches_independent_finite_differences(self):
        """Every producer row agrees with direct commutator finite differences."""
        t = cauchy_family(
            [Q(-2), Q(0), Q(3), Q(7)],
            [Q(1), Q(-2), Q(3), Q(5)],
            Q(4),
        )
        h = off_diagonal(t)
        independent_h = decode({'n': 4, 'orbits': to_orbits(h)})
        for key, row, rhs in equations(h):
            self.assertEqual((row, rhs), _row(independent_h, key))

    def test_nonuniform_cauchy_orientation_and_cocycle(self):
        """The reverse-pair ratio and Cauchy cocycle hold beyond all-ones directions."""
        nodes = [Q(-3), Q(0), Q(2), Q(9)]
        z = [Q(2), Q(-3), Q(5), Q(7)]
        t = cauchy_family(nodes, z, Q(11, 3))
        h = off_diagonal(t)
        cert = solve(h)
        self.assertEqual(cert['status'], 'ambiguous')
        self.assertEqual(cert['rank'], 3)
        self.assertTrue(verify({'n': 4, 'orbits': to_orbits(h)}, cert))

        null = list(map(Q, cert['null_basis'][0]))
        scale = null[0] / z[0]
        self.assertEqual(null, [scale * value for value in z])
        for i, j in combinations(range(4), 2):
            self.assertEqual(h[j][j][i], -(z[j] / z[i]) * h[i][i][j])
        for i, j, k in combinations(range(4), 3):
            d_ij = 1 / (z[j] * h[i][i][j])
            d_jk = 1 / (z[k] * h[j][j][k])
            d_ik = 1 / (z[k] * h[i][i][k])
            self.assertEqual(d_ij + d_jk, d_ik)

    def test_full_rank_distance_spectrum_at_dimension_nine(self):
        """Every distance 1..n is realized by a full-mode-rank odeco tensor."""
        n = 9
        for m in range(1, n + 1):
            t = distance_family(n, m, Q(2))
            h = off_diagonal(t)
            cert = solve(h)
            graph = kernel_graph(h)
            self.assertEqual(cert['status'], 'ambiguous')
            self.assertEqual(cert['rank'], n - 1)
            self.assertEqual(len(graph['null_basis']), 1)
            support = [i for i, value in enumerate(map(Q, graph['null_basis'][0])) if value]
            self.assertEqual(len(support), m)
            unfolding = [[t[i][j][k] for j in range(n) for k in range(n)] for i in range(n)]
            self.assertEqual(_rank(unfolding), n)

    def test_conditional_radius_contains_a_compatible_truth(self):
        """A nonzero same-fiber candidate is enclosed under one gross diagonal error."""
        nodes = [Q(0), Q(1), Q(3)]
        z = [Q(1), Q(-2), Q(3)]
        truth_tensor = cauchy_family(nodes, z, Q(1))
        candidate_tensor = cauchy_family(nodes, z, Q(8, 7))
        h = off_diagonal(truth_tensor)
        truth = [truth_tensor[i][i][i] for i in range(3)]
        candidate = [candidate_tensor[i][i][i] for i in range(3)]
        observed = truth.copy()
        observed[0] += Q(10**20)

        cert = certify(h, observed, candidate, [0], 1, 0, 0)
        self.assertTrue(verify_stability(cert))
        error_sq = sum(((candidate[i] - truth[i]) ** 2 for i in range(3)), Q(0))
        self.assertLessEqual(error_sq, Q(cert['diagonal_radius']) ** 2)
        self.assertLessEqual(error_sq, Q(cert['tensor_radius']) ** 2)


if __name__ == '__main__':
    unittest.main()
