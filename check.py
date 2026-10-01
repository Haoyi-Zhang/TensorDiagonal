#!/usr/bin/env python3
"""Recheck saved scientific records; do not trust producer acceptance flags."""
from __future__ import annotations

import argparse
import csv
import json
import random
import resource
import sys
import time
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations, combinations_with_replacement, permutations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from verify import verify, verify_stability, decode, _comm, _rank, _row

EXACT_SPECS = {
    'ternary': (3, (-1, 0, 1), True),
    'pair_grid': (3, (-2, -1, 0, 1, 2), False),
    'binary4': (4, (0, 1), True),
}
EXACT_EXPECTED = {
    'ternary': Counter({'unique': 444, 'ambiguous': 73, 'inconsistent': 1670}),
    'pair_grid': Counter({'unique': 0, 'ambiguous': 469, 'inconsistent': 15156}),
    'binary4': Counter({'unique': 297, 'ambiguous': 230, 'inconsistent': 65009}),
}
NOISE_DIMENSION_BUDGETS = ((3, 1), (5, 1), (7, 2))
NOISE_POWERS = (10, 14, 18)
NOISE_AMPLITUDES = (1, 1 << 10, 1 << 20, 1 << 40)


def load(path: Path):
    return json.loads(path.read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def zero_tensor(n):
    return [[[Q(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]


def tensor_orbits(t):
    return [str(t[i][j][k]) for i, j, k in combinations_with_replacement(range(len(t)), 3)]


def off_diagonal_tensor(t):
    h = [[row[:] for row in sl] for sl in t]
    for i in range(len(h)):
        h[i][i][i] = Q(0)
    return h


def full_tensor(n, orbits):
    # decode intentionally removes the diagonal; restore it explicitly.
    t = decode({'n': n, 'orbits': orbits})
    for key, value in zip(combinations_with_replacement(range(n), 3), orbits):
        if key[0] == key[2]:
            t[key[0]][key[0]][key[0]] = Q(value)
    return t


def independent_cauchy_tensor(nodes, direction, parameter):
    nodes = list(map(Q, nodes))
    z = list(map(Q, direction))
    parameter = Q(parameter)
    n = len(nodes)
    require(len(z) == n and len(set(nodes)) == n and all(z), 'invalid independent Cauchy design')
    t = zero_tensor(n)
    for i in range(n):
        t[i][i][i] = z[i] * (
            parameter
            - sum((1 / (z[k] ** 2 * (nodes[k] - nodes[i])) for k in range(n) if k != i), Q(0))
        )
        for j in range(n):
            if i != j:
                value = 1 / (z[j] * (nodes[j] - nodes[i]))
                t[i][i][j] = t[i][j][i] = t[j][i][i] = value
    return t


def independent_householder(vector):
    v = list(map(Q, vector))
    n = len(v)
    norm = sum((x * x for x in v), Q(0))
    require(norm > 0, 'zero Householder vector')
    u = [[Q(i == j) - 2 * v[i] * v[j] / norm for j in range(n)] for i in range(n)]
    require(
        all(
            sum((u[i][a] * u[i][b] for i in range(n)), Q(0)) == Q(a == b)
            for a in range(n)
            for b in range(n)
        ),
        'nonorthogonal Householder factor',
    )
    return u


def independent_synthesize(u, weights):
    weights = list(map(Q, weights))
    n = len(u)
    require(len(weights) == n and all(len(row) == n for row in u), 'factor/weight dimensions')
    return [
        [
            [sum((weights[a] * u[i][a] * u[j][a] * u[k][a] for a in range(n)), Q(0)) for k in range(n)]
            for j in range(n)
        ]
        for i in range(n)
    ]


def householder_tensor(vector, weights):
    return independent_synthesize(independent_householder(vector), weights)


def independent_rotate_tensor(t, rotation):
    n = len(t)
    a = [
        [[sum((rotation[i][p] * t[p][j][k] for p in range(n)), Q(0)) for k in range(n)] for j in range(n)]
        for i in range(n)
    ]
    b = [
        [[sum((rotation[j][p] * a[i][p][k] for p in range(n)), Q(0)) for k in range(n)] for j in range(n)]
        for i in range(n)
    ]
    return [
        [[sum((rotation[k][p] * b[i][j][p] for p in range(n)), Q(0)) for k in range(n)] for j in range(n)]
        for i in range(n)
    ]


def independent_distance_tensor(n, m, parameter=Q(1)):
    require(1 <= m <= n <= 16, 'invalid distance design')
    if m == n:
        return independent_cauchy_tensor(range(n), [Q(1)] * n, parameter)
    small = independent_cauchy_tensor(range(m), [Q(1)] * m, parameter)
    t = zero_tensor(n)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                t[i][j][k] = small[i][j][k]
        t[i][i][i] -= i
        t[i][i][m] = t[i][m][i] = t[m][i][i] = Q(1)
    t[m][m][m] = Q(1)
    for i in range(m + 1, n):
        t[i][i][i] = Q(i - m + 1)
    q = n - m
    complement = independent_householder(range(1, q + 1))
    rotation = [[Q(i == j) for j in range(n)] for i in range(n)]
    for i in range(q):
        for j in range(q):
            rotation[m + i][m + j] = complement[i][j]
    return independent_rotate_tensor(t, rotation)


def independent_rows(h):
    n = len(h)
    keys = [(i, j, p, q) for i, j in combinations(range(n), 2) for p, q in combinations(range(n), 2)]
    return [_row(h, key) for key in keys]


def independent_particular(h):
    """Return a zero-free-coordinate particular solution using fresh RREF code."""
    n = len(h)
    augmented = [row + [rhs] for row, rhs in independent_rows(h)]
    rank = 0
    pivots = []
    for column in range(n):
        pivot = next((k for k in range(rank, len(augmented)) if augmented[k][column]), None)
        if pivot is None:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        divisor = augmented[rank][column]
        augmented[rank] = [value / divisor for value in augmented[rank]]
        for k in range(len(augmented)):
            if k != rank and augmented[k][column]:
                multiplier = augmented[k][column]
                augmented[k] = [x - multiplier * y for x, y in zip(augmented[k], augmented[rank])]
        pivots.append(column)
        rank += 1
    require(
        all(any(row[:n]) or not row[n] for row in augmented),
        'independent affine solver found an inconsistent fixed fiber',
    )
    x = [Q(0)] * n
    for row_index, column in enumerate(pivots):
        x[column] = augmented[row_index][n]
    require(
        all(sum((a * b for a, b in zip(row, x)), Q(0)) == rhs for row, rhs in independent_rows(h)),
        'independent particular solution does not satisfy the affine system',
    )
    return x, rank


def independent_free_components(h, verify_nullity=True):
    """Recover free gain components without using a saved null basis."""
    n = len(h)
    pinned = set()
    adjacency = [[] for _ in range(n)]
    for i, j, k in combinations(range(n), 3):
        if h[i][j][k]:
            pinned.update((i, j, k))
    for i, j in combinations(range(n), 2):
        a = h[i][i][j]
        b = h[i][j][j]
        if a and b:
            adjacency[i].append((j, -b / a))
            adjacency[j].append((i, -a / b))
        elif b:
            pinned.add(i)
        elif a:
            pinned.add(j)

    seen = set()
    free = []
    for start in range(n):
        if start in seen:
            continue
        weights = {start: Q(1)}
        stack = [start]
        seen.add(start)
        balanced = True
        while stack:
            i = stack.pop()
            for j, gain in adjacency[i]:
                desired = weights[i] * gain
                if j in weights:
                    if weights[j] != desired:
                        balanced = False
                else:
                    weights[j] = desired
                    seen.add(j)
                    stack.append(j)
        support = tuple(sorted(weights))
        if balanced and not (set(support) & pinned):
            for i, j in combinations(support, 2):
                require(h[i][i][j] and h[i][j][j], 'free support is not a complete clique')
                require(weights[j] == weights[i] * (-h[i][j][j] / h[i][i][j]), 'free-support gain mismatch')
            for triple in combinations(range(n), 3):
                if set(support).intersection(triple):
                    require(h[triple[0]][triple[1]][triple[2]] == 0, 'all-distinct entry pins a free support')
            direction = [Q(0)] * n
            for i, value in weights.items():
                direction[i] = value
            free.append({'support': support, 'direction': direction})

    used = set()
    for component in free:
        support = set(component['support'])
        require(support and not (used & support), 'free component supports overlap')
        used.update(support)

    if not verify_nullity:
        return free, None
    rows = [row for row, _ in independent_rows(h)]
    rank = _rank(rows)
    directions = [component['direction'] for component in free]
    require(len(free) == n - rank, 'free-component count does not equal affine nullity')
    require(_rank(directions) == len(directions), 'free-component directions are dependent')
    require(
        all(sum((a * b for a, b in zip(row, z)), Q(0)) == 0 for z in directions for row in rows),
        'free-component direction is not in the affine kernel',
    )
    return free, rank


def support_signature(record):
    """Independently check the disjoint-support/clique form of a saved fiber."""
    cert = record['certificate']
    if cert['status'] == 'inconsistent':
        return None
    h = decode(record['input'])
    free, _ = independent_free_components(h, verify_nullity=False)
    require(len(free) == len(cert['null_basis']), 'certificate nullity disagrees with independent component count')
    sizes = [len(component['support']) for component in free]
    return ','.join(map(str, sorted(sizes))) if sizes else 'rigid'


def check_declared_design(design):
    require(design['distance_realizations']['n_min'] == 1, 'declared distance n_min')
    require(design['distance_realizations']['n_max'] == 8, 'declared distance n_max')
    require(design['distance_realizations']['distances'] == 'all integers 1 through n', 'declared distance range')
    require(design['distance_realizations']['instances'] == 36, 'declared distance count')
    require(design['cauchy_family']['n_min'] == 2 and design['cauchy_family']['n_max'] == 8, 'declared Cauchy range')
    require(design['cauchy_family']['parameters'] == [0, 1, 2], 'declared Cauchy parameters')
    require(len(design['cauchy_family']['directions']) == 2 and design['cauchy_family']['instances'] == 42,
            'declared Cauchy design count')
    householder = design['householder_recovery']
    require(householder['n_min'] == 3 and householder['n_max'] == 8 and householder['instances'] == 6,
            'declared Householder design')
    generic = design['generic_rigid_witnesses']
    require(generic['n_min'] == 3 and generic['n_max'] == 12 and generic['instances'] == 10,
            'declared generic-rigidity design')
    require(design['structured_expected_instances'] == 94, 'declared structured total')
    decoding = design['decoding']
    require(
        decoding['fixed_fibers'] == 2
        and decoding['first_dimension'] == 3
        and decoding['first_free_supports'] == [[0, 1, 2]]
        and decoding['second_dimension'] == 5
        and decoding['second_free_supports'] == [[0, 1], [2, 3], [4]]
        and decoding['observation_alphabet'] == [-1, 0, 1]
        and decoding['words'] == 270
        and decoding['budget_checks'] == 1566,
        'declared decoding design',
    )
    noise = design['noise']
    require([tuple(pair) for pair in noise['dimension_budget_pairs']] == list(NOISE_DIMENSION_BUDGETS),
            'declared noise dimensions/budgets')
    require(tuple(noise['noise_exponents']) == NOISE_POWERS, 'declared noise powers')
    require(tuple(noise['amplitudes']) == NOISE_AMPLITUDES, 'declared noise amplitudes')
    require(
        noise['distinct_dimension_noise_configurations'] == 9
        and noise['gross_amplitudes_per_configuration'] == 4
        and noise['instances'] == 36,
        'declared noise configuration count',
    )
    require(design['resource_limits']['implementation_max_n'] == 16, 'declared implementation dimension cap')


def structured_design():
    design = {}
    for n in range(1, 9):
        for m in range(1, n + 1):
            case = f'distance-{n}-{m}'
            design[case] = {
                'family': 'distance',
                'n': n,
                'parameter': m,
                'truth': independent_distance_tensor(n, m),
                'tensor_rank': n,
                'fiber_dim': 1,
                'distance': m,
            }
    counter = 36
    for n in range(2, 9):
        for direction_index, direction in enumerate(
            ([Q(1)] * n, [Q((-1) ** i * (i + 1)) for i in range(n)])
        ):
            for kappa in (0, 1, 2):
                case = f'cauchy-{n}-{counter}'
                design[case] = {
                    'family': 'cauchy',
                    'n': n,
                    'parameter': (direction_index, kappa),
                    'truth': independent_cauchy_tensor(range(n), direction, kappa),
                    'tensor_rank': n - int(kappa == 0),
                    'fiber_dim': 1,
                    'distance': n,
                    'kappa': kappa,
                }
                counter += 1
    for n in range(3, 9):
        case = 'householder-three-coordinate-ambiguity' if n == 3 else f'rigid-{n}'
        design[case] = {
            'family': 'householder',
            'n': n,
            'parameter': tuple(range(1, n + 1)),
            'truth': householder_tensor(range(1, n + 1), range(1, n + 1)),
            'tensor_rank': n,
            'fiber_dim': 1 if n == 3 else 0,
            'distance': 3 if n == 3 else None,
        }
    for n in range(3, 13):
        case = f'generic-rigid-{n}'
        design[case] = {
            'family': 'generic-rigid',
            'n': n,
            'parameter': (tuple(range(1, n + 1)), tuple(1 << i for i in range(n))),
            'truth': householder_tensor(range(1, n + 1), [1 << i for i in range(n)]),
            'tensor_rank': n,
            'fiber_dim': 0,
            'distance': None,
        }
    require(len(design) == 94, 'internal structured design count')
    return design


def check_structured_records(structured, summary=None, design=None):
    require(isinstance(structured, list) and len(structured) == 94, 'structured case count')
    cases = [record.get('case') for record in structured]
    require(all(isinstance(case, str) for case in cases), 'structured case identifier')
    require(len(set(cases)) == len(cases), 'duplicate structured case')
    design = structured_design() if design is None else design
    require(set(cases) == set(design), 'structured case set does not match declared design')

    for record in structured:
        case = record['case']
        spec = design[case]
        n = spec['n']
        require(record['input']['n'] == n, 'structured dimension mismatch: ' + case)
        expected_truth = tensor_orbits(spec['truth'])
        expected_input = tensor_orbits(off_diagonal_tensor(spec['truth']))
        require([Q(v) for v in record['truth']] == [Q(v) for v in expected_truth], 'structured truth/design mismatch: ' + case)
        require(
            [Q(v) for v in record['input']['orbits']] == [Q(v) for v in expected_input],
            'structured input/design mismatch: ' + case,
        )
        require(
            all(
                Q(observed) == Q(truth)
                for key, observed, truth in zip(
                    combinations_with_replacement(range(n), 3), record['input']['orbits'], record['truth']
                )
                if key[0] != key[2]
            ),
            'structured truth mixed orbits are not bound to certificate input: ' + case,
        )
        require(verify(record['input'], record['certificate']), 'structured certificate: ' + case)
        t = full_tensor(n, record['truth'])
        commutator_keys = [(i, j, p, q) for i, j in combinations(range(n), 2) for p, q in combinations(range(n), 2)]
        require(not any(_comm(t, key) for key in commutator_keys), 'truth is not commuting: ' + case)
        tensor_rank = _rank([[t[i][j][k] for j in range(n) for k in range(n)] for i in range(n)])
        require(tensor_rank == record['tensor_rank'] == spec['tensor_rank'], 'structured tensor rank: ' + case)

        h = decode(record['input'])
        free, affine_rank = independent_free_components(h)
        fiber_dim = n - affine_rank
        require(fiber_dim == spec['fiber_dim'], 'structured fiber dimension: ' + case)
        require(record['certificate']['rank'] == affine_rank, 'structured certificate rank: ' + case)
        distance = min((len(component['support']) for component in free), default=None)
        require(distance == spec['distance'], 'structured component distance: ' + case)
        if spec['family'] == 'cauchy':
            if 'distance' in record:
                require(record['distance'] == spec['distance'], 'structured reported distance: ' + case)
        else:
            require(record.get('distance') == spec['distance'], 'structured reported distance: ' + case)

        if spec['family'] == 'cauchy':
            require(record.get('kappa') == spec['kappa'], 'structured Cauchy parameter: ' + case)
        if spec['family'] == 'generic-rigid':
            vector, weights = spec['parameter']
            require(list(map(Q, record['householder_vector'])) == list(map(Q, vector)), 'generic witness vector: ' + case)
            require(list(map(Q, record['weights'])) == list(map(Q, weights)), 'generic witness weights: ' + case)
            triples = [t[i][j][k] for i, j, k in combinations(range(n), 3)]
            require(triples and all(triples), 'generic witness has a zero all-distinct entry: ' + case)
            require(record['all_distinct_nonzero'] == len(triples), 'generic witness triple count: ' + case)
            require(Q(record['minimum_all_distinct_magnitude']) == min(map(abs, triples)), 'generic witness magnitude: ' + case)
            require(
                list(map(Q, record['certificate']['particular'])) == [t[i][i][i] for i in range(n)],
                'generic witness diagonal: ' + case,
            )

    if summary is not None:
        expected = {
            'instances': 94,
            'distance_cases': 36,
            'cauchy_cases': 42,
            'rigid_cases': 5,
            'additional_ambiguous_case': 1,
            'generic_rigid_witnesses': 10,
            'failures': 0,
        }
        require(all(summary.get(key) == value for key, value in expected.items()), 'structured summary mismatch')


def noise_key(case):
    require(isinstance(case, str), 'noise case identifier')
    parts = case.split('-')
    require(len(parts) == 5 and parts[0] == 'noise', 'noncanonical noise case identifier')
    try:
        values = tuple(int(value) for value in parts[1:])
    except ValueError as exc:
        raise ValueError('noncanonical noise case identifier') from exc
    require(case == 'noise-' + '-'.join(map(str, values)), 'noncanonical noise case identifier')
    return values


def expected_noise_case(n, s, power, amplitude):
    truth = independent_distance_tensor(n, n)
    h = off_diagonal_tensor(truth)
    rng = random.Random(907 + n)
    perturbation = zero_tensor(n)
    for i, j, k in combinations_with_replacement(range(n), 3):
        value = Q(0) if i == k else Q(rng.randint(-2, 2), 1 << power)
        for a, b, c in set(permutations((i, j, k))):
            perturbation[a][b][c] = value
    diagonal_noise = [Q(rng.randint(-2, 2), 1 << power) for _ in range(n)]
    mixed = [
        [[h[i][j][k] + perturbation[i][j][k] for k in range(n)] for j in range(n)]
        for i in range(n)
    ]
    observed = [
        truth[i][i][i] + diagonal_noise[i] + (Q(amplitude * (i + 1)) if i < s else Q(0))
        for i in range(n)
    ]
    return truth, mixed, observed


def check_noise_records(noise, csv_rows, summary=None):
    expected_keys = {
        (n, s, power, amplitude)
        for n, s in NOISE_DIMENSION_BUDGETS
        for power in NOISE_POWERS
        for amplitude in NOISE_AMPLITUDES
    }
    require(isinstance(noise, list) and len(noise) == 36, 'noise case count')
    noise_cases = [record.get('case') for record in noise]
    require(len(set(noise_cases)) == len(noise_cases), 'duplicate noise certificate case')
    require(isinstance(csv_rows, list) and len(csv_rows) == 36, 'noise csv count')
    csv_cases = [row.get('case') for row in csv_rows]
    require(len(set(csv_cases)) == len(csv_cases), 'duplicate noise csv case')
    cert_keys = {noise_key(case) for case in noise_cases}
    table_keys = {noise_key(case) for case in csv_cases}
    require(cert_keys == table_keys == expected_keys, 'noise certificate/CSV/design key set mismatch')
    by_case = {record['case']: record for record in noise}
    csv_by_case = {row['case']: row for row in csv_rows}

    amplitude_groups = {}
    for key in sorted(expected_keys):
        n, s, power, amplitude = key
        case = f'noise-{n}-{s}-{power}-{amplitude}'
        record = by_case[case]
        row = csv_by_case[case]
        cert = record['certificate']
        require(verify_stability(cert), 'invalid posterior witness ' + case)
        require(cert['n'] == n and cert['corruption_budget'] == s, 'noise certificate n/s mismatch: ' + case)
        require(
            int(row['n']) == n
            and int(row['s']) == s
            and int(row['noise_exponent']) == power
            and int(row['amplitude']) == amplitude,
            'noise CSV n/s/p/amplitude mismatch: ' + case,
        )

        expected_truth, expected_mixed, expected_observed = expected_noise_case(n, s, power, amplitude)
        require([Q(v) for v in record['truth']] == [Q(v) for v in tensor_orbits(expected_truth)], 'noise truth/design mismatch: ' + case)
        require([Q(v) for v in cert['mixed_orbits']] == [Q(v) for v in tensor_orbits(expected_mixed)], 'noise mixed/design mismatch: ' + case)
        require(list(map(Q, cert['observed_diagonal'])) == expected_observed, 'noise observation/design mismatch: ' + case)

        t = full_tensor(n, record['truth'])
        h = decode({'n': n, 'orbits': cert['mixed_orbits']})
        truth_diagonal = [t[i][i][i] for i in range(n)]
        candidate = list(map(Q, cert['candidate_diagonal']))
        error = sum((a - b) ** 2 for a, b in zip(candidate, truth_diagonal))
        require(error == Q(record['actual_error_squared']) == Q(row['error_squared']), 'actual error mismatch: ' + case)
        require(error <= Q(cert['diagonal_radius']) ** 2, 'truth outside radius: ' + case)
        mixed_error = sum(
            (h[i][j][k] - t[i][j][k]) ** 2
            for i in range(n)
            for j in range(n)
            for k in range(n)
            if not i == j == k
        )
        require(mixed_error <= Q(cert['delta']) ** 2, 'declared mixed bound violated: ' + case)
        observed = list(map(Q, cert['observed_diagonal']))
        require(
            sum((observed[i] - truth_diagonal[i]) ** 2 for i in range(s, n)) <= Q(cert['epsilon']) ** 2,
            'clean diagonal bound violated: ' + case,
        )
        commutator_keys = [(i, j, p, q) for i, j in combinations(range(n), 2) for p, q in combinations(range(n), 2)]
        require(not any(_comm(t, item) for item in commutator_keys), 'non-odeco noisy truth: ' + case)
        for field in ('delta', 'epsilon', 'gamma', 'diagonal_radius', 'tensor_radius'):
            require(Q(row[field]) == Q(cert[field]), 'csv/certificate discrepancy: ' + case + ':' + field)
        require(row['selected_support'] == ';'.join(map(str, cert['candidate_support'])), 'noise selected support mismatch: ' + case)
        require(row['verified'] == 'True', 'noise CSV verified flag mismatch: ' + case)
        require(row['same_as_unit_amplitude'] == 'True', 'noise amplitude flag mismatch: ' + case)

        group_key = (n, s, power)
        value = (cert['candidate_diagonal'], cert['candidate_support'], cert['diagonal_radius'])
        if group_key in amplitude_groups:
            require(value == amplitude_groups[group_key], 'amplitude sensitivity differs from reported result')
        else:
            amplitude_groups[group_key] = value
    require(len(amplitude_groups) == 9, 'noise configuration count')

    if summary is not None:
        expected = {
            'instances': 36,
            'verified': 36,
            'same_as_unit_amplitude': 36,
            'negative_controls': 2,
            'failures': 0,
        }
        require(all(summary.get(key) == value for key, value in expected.items()), 'noise summary mismatch')


def fixed_decoding_fibers():
    first = independent_cauchy_tensor([Q(0), Q(1), Q(2)], [Q(1)] * 3, Q(1))
    left = independent_cauchy_tensor([Q(0), Q(1)], [Q(1), Q(2)], Q(1))
    right = independent_cauchy_tensor([Q(0), Q(2)], [Q(1), Q(1)], Q(1))
    second = zero_tensor(5)
    for i, j, k in product(range(2), repeat=3):
        second[i][j][k] = left[i][j][k]
        second[i + 2][j + 2][k + 2] = right[i][j][k]
    second[4][4][4] = Q(3)
    return (first, second)


def independent_list_sizes_from_geometry(x, free, observed):
    n = len(x)
    used = set().union(*(set(component['support']) for component in free)) if free else set()
    rigid_cost = sum(observed[i] != x[i] for i in range(n) if i not in used)
    choices = []
    for component in free:
        support = component['support']
        direction = component['direction']
        observed_parameters = sorted({(observed[i] - x[i]) / direction[i] for i in support})
        finite = []
        for alpha in observed_parameters:
            cost = sum(x[i] + alpha * direction[i] != observed[i] for i in support)
            finite.append((cost, False))
        choices.append(finite + [(len(support), True)])

    list_sizes = []
    for budget in range(n + 1):
        finite_count = 0
        infinite = False
        for selection in product(*choices):
            cost = rigid_cost + sum(value for value, _ in selection)
            if cost <= budget:
                if any(unseen for _, unseen in selection):
                    infinite = True
                else:
                    finite_count += 1
        list_sizes.append('infinite' if infinite else finite_count)
    return list_sizes


def independent_list_sizes(h, observed):
    x, rank = independent_particular(h)
    free, component_rank = independent_free_components(h)
    require(rank == component_rank, 'decoding affine-rank disagreement')
    return x, free, independent_list_sizes_from_geometry(x, free, observed)


def check_decoding_records(payload, summary=None):
    require(isinstance(payload, dict) and isinstance(payload.get('cases'), list), 'decoding payload')
    records = payload['cases']
    require(len(records) == 270, 'decoding count')
    case_ids = [record.get('case') for record in records]
    require(len(set(case_ids)) == len(case_ids), 'duplicate decoding case')
    by_case = {record['case']: record for record in records}

    expected_ids = set()
    checks = 0
    for case_index, tensor in enumerate(fixed_decoding_fibers()):
        h = off_diagonal_tensor(tensor)
        x, rank = independent_particular(h)
        free, component_rank = independent_free_components(h)
        require(rank == component_rank, 'decoding affine-rank disagreement')
        supports = [component['support'] for component in free]
        expected_supports = [(0, 1, 2)] if case_index == 0 else [(0, 1), (2, 3), (4,)]
        require(supports == expected_supports, 'fixed decoding fiber geometry')
        n = len(tensor)
        for word_index, word in enumerate(product((-1, 0, 1), repeat=n)):
            case = f'word-{case_index}-{word_index}'
            expected_ids.add(case)
            require(case in by_case, 'missing decoding case: ' + case)
            record = by_case[case]
            observed = [x[i] + Q(word[i]) for i in range(n)]
            list_sizes = independent_list_sizes_from_geometry(x, free, observed)
            checks += len(list_sizes)
            require(list(map(Q, record['observed_diagonal'])) == observed, 'decoding observation mismatch: ' + case)
            require(record['list_sizes'] == list_sizes, 'decoding budget list mismatch: ' + case)
            minimum_errors = next(i for i, value in enumerate(list_sizes) if value != 0)
            minimum_list_size = list_sizes[minimum_errors]
            infinite_at_budget = next((i for i, value in enumerate(list_sizes) if value == 'infinite'), None)
            require(minimum_list_size != 'infinite', 'fixed decoding design has infinite minimum list')
            require(record['minimum_errors'] == minimum_errors, 'minimum list budget: ' + case)
            require(record['minimum_list_size'] == minimum_list_size, 'minimum list size: ' + case)
            require(record['infinite_at_budget'] == infinite_at_budget, 'infinite threshold: ' + case)
    require(set(case_ids) == expected_ids, 'decoding case set mismatch')
    require(checks == 1566, 'budget count')

    ablation = payload.get('normalization_ablation')
    require(isinstance(ablation, dict), 'normalization ablation missing')
    direction = list(map(Q, ablation['direction']))
    ratios = list(map(Q, ablation['ratios']))
    require(direction == [Q(100), Q(1), Q(1)] and ratios == [Q(2), Q(1), Q(1)], 'normalization ablation design')
    costs = {str(alpha): str(sum((abs(z * (alpha - ratio)) for z, ratio in zip(direction, ratios)), Q(0))) for alpha in set(ratios)}
    require(ablation['l1_costs'] == costs, 'normalization ablation costs')
    require(ablation['true_parameter'] == '1' and ablation['modal_parameter'] == '1' and ablation['unnormalized_l1_parameter'] == '2',
            'normalization ablation labels')

    if summary is not None:
        expected = {'words': 270, 'budget_checks': 1566, 'normalization_cases': 1, 'failures': 0}
        require(all(summary.get(key) == value for key, value in expected.items()), 'decoding summary mismatch')


def check(results):
    check_declared_design(load(ROOT / 'inputs' / 'design.json'))
    counts = Counter()
    families = {}
    profiles = {}
    family_indices = {name: set() for name in EXACT_SPECS}
    seen = set()
    maxbits = 0
    with (results / 'exhaustive.jsonl').open() as handle:
        for line in handle:
            record = json.loads(line)
            require(record['case'] not in seen, 'duplicate finite case')
            seen.add(record['case'])
            require(verify(record['input'], record['certificate']), 'invalid exact certificate ' + record['case'])
            status = record['certificate']['status']
            counts[status] += 1
            family, index_text = record['case'].rsplit('-', 1)
            index = int(index_text)
            require(family in EXACT_SPECS, 'unknown finite family')
            n, alphabet, include_all_distinct = EXACT_SPECS[family]
            keys = list(combinations_with_replacement(range(n), 3))
            selected = [key for key in keys if key[0] != key[-1] and (include_all_distinct or len(set(key)) < 3)]
            total = len(alphabet) ** len(selected)
            require(0 <= index < total, 'out-of-range case index')
            require(index not in family_indices[family], 'duplicate finite-family index')
            family_indices[family].add(index)
            require(record['case'] == f'{family}-{index:05d}', 'noncanonical case identifier')
            number = index
            digit_indices = [0] * len(selected)
            for position in range(len(selected) - 1, -1, -1):
                number, remainder = divmod(number, len(alphabet))
                digit_indices[position] = remainder
            require(number == 0, 'case index decoding overflow')
            expected = {key: alphabet[digit_indices[position]] for position, key in enumerate(selected)}
            require(
                record['input']['n'] == n
                and [Q(value) for value in record['input']['orbits']] == [expected.get(key, 0) for key in keys],
                'case does not match declared Cartesian design',
            )
            families.setdefault(family, Counter())[status] += 1
            signature = support_signature(record)
            if signature is not None:
                profiles.setdefault(family, Counter())[signature] += 1
                certificate = record['certificate']
                for value in certificate['particular'] + sum(certificate['null_basis'], []):
                    rational = Q(value)
                    maxbits = max(maxbits, abs(rational.numerator).bit_length(), rational.denominator.bit_length())
    require(families == EXACT_EXPECTED, 'finite family classification mismatch')
    for family, (n, alphabet, include_all_distinct) in EXACT_SPECS.items():
        keys = list(combinations_with_replacement(range(n), 3))
        selected = [key for key in keys if key[0] != key[-1] and (include_all_distinct or len(set(key)) < 3)]
        require(family_indices[family] == set(range(len(alphabet) ** len(selected))), 'incomplete Cartesian family')
    require(counts == Counter({'unique': 741, 'ambiguous': 772, 'inconsistent': 81835}), 'finite classification mismatch')
    summary = load(results / 'exhaustive-summary.json')
    require(
        summary['instances'] == 83348
        and set(summary['counts']) == set(families)
        and all(Counter(summary['counts'][key]) == value for key, value in families.items()),
        'finite summary mismatch',
    )
    require(
        set(summary['consistent_support_profiles']) == set(profiles)
        and all(Counter(summary['consistent_support_profiles'][key]) == value for key, value in profiles.items()),
        'finite support-profile summary mismatch',
    )
    require(summary['maximum_completion_coefficient_bits'] == maxbits, 'finite coefficient-bit summary mismatch')
    del seen, family_indices, summary

    check_structured_records(load(results / 'structured.json'), load(results / 'structured-summary.json'))

    noise = load(results / 'noise-certificates.json')
    with (results / 'noise.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    check_noise_records(noise, csv_rows, load(results / 'noise-summary.json'))
    del noise, csv_rows

    check_decoding_records(load(results / 'decoding.json'), load(results / 'decoding-summary.json'))

    negative = load(results / 'negative-controls.json')
    require(negative['distance_two_budget_one']['reason'] == 'singular_restricted_system', 'singular control')
    require(negative['excessive_noise_bound']['reason'] == 'noise_exceeds_certified_margin', 'noise-margin control')
    mutation = load(results / 'mutations.json')
    require(len(mutation) == 10 and all(record['rejected'] for record in mutation), 'mutation records')
    for suite in ('exhaustive', 'structured', 'decoding', 'noise', 'mutations'):
        require(load(results / (suite + '-summary.json'))['failures'] == 0, 'reported failure')
    return {
        'exact_inputs': 83348,
        'structured_inputs': 94,
        'noisy_certificates': 36,
        'decoding_words': 270,
        'budget_records': 1566,
        'mutation_records': 10,
        'status': 'verified',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    cpu = time.process_time()
    wall = time.perf_counter()
    report = check(args.results)
    report.update(
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.perf_counter() - wall,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
