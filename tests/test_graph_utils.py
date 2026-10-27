"""Independent small-graph oracles for targets, CPDAGs and decoders."""

import itertools
import unittest
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_utils_BAM import (
    collider_targets, dag_to_cpdag, decode_cpdag, decode_dag,
    skeleton_from_dag, validate_dag,
)


def oracle_is_dag(a):
    reachable = np.asarray(a, dtype=bool).copy()
    for k in range(len(a)):
        reachable |= reachable[:, k, None] & reachable[None, k, :]
    return not np.diag(reachable).any()


def oracle_colliders(a):
    result = []
    for center in range(len(a)):
        for first, last in itertools.combinations(range(len(a)), 2):
            if a[first, center] and a[last, center] and not (a[first, last] or a[last, first]):
                result.append((first, center, last))
    return tuple(result)


def all_dags(d):
    pairs = list(itertools.combinations(range(d), 2))
    for states in itertools.product(range(3), repeat=len(pairs)):
        a = np.zeros((d, d), dtype=np.uint8)
        for (i, j), state in zip(pairs, states):
            if state == 1:
                a[i, j] = 1
            elif state == 2:
                a[j, i] = 1
        if oracle_is_dag(a):
            yield a


def dag_for_order(s, order):
    a = np.zeros_like(s)
    for pos, source in enumerate(order):
        for target in order[pos + 1:]:
            a[source, target] = s[source, target]
    return a


def collider_score(a, probabilities):
    s = a | a.T
    value = 0.0
    for j in range(len(a)):
        for i, k in itertools.combinations(range(len(a)), 2):
            if s[i, j] and s[k, j] and not s[i, k]:
                p = np.clip((probabilities[i, j, k] + probabilities[k, j, i]) / 2,
                            1e-7, 1 - 1e-7)
                value += np.log(p if a[i, j] and a[k, j] else 1 - p)
    return value


class GraphUtilityTests(unittest.TestCase):
    def test_targets_use_source_target_convention(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        a[0, 1] = a[2, 1] = a[1, 3] = 1
        labels, mask = collider_targets(a)
        self.assertEqual(labels[0, 1, 2], 1)
        self.assertEqual(labels[2, 1, 0], 1)
        self.assertEqual(labels[0, 1, 3], 0)
        self.assertTrue(mask[0, 1, 3])
        self.assertFalse(mask[0, 1, 0])
        self.assertEqual(int(labels.sum()), 2)
        np.testing.assert_array_equal(labels, labels.transpose(2, 1, 0))
        a[0, 2] = 1  # Shielding removes the original collider target.
        labels, mask = collider_targets(a)
        self.assertFalse(mask[0, 1, 2])
        self.assertFalse(labels[0, 1, 2])

    def test_validation_and_no_input_mutation(self):
        a = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        original = a.copy()
        copy = validate_dag(a)
        copy[0, 1] = 0
        for operation in (skeleton_from_dag, collider_targets, dag_to_cpdag):
            operation(a)
        np.testing.assert_array_equal(a, original)
        for bad in (np.ones((2, 3)), np.eye(2), [[0, 1], [1, 0]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    [[0, np.nan], [0, 0]], [[0, 0.5], [0, 0]]):
            with self.assertRaises(ValueError):
                validate_dag(bad)

    def test_exact_cpdag_for_every_four_node_dag(self):
        # Independently enumerate all 543 labeled DAGs, group by skeleton and
        # unshielded colliders, then take the union of arrows within each class.
        # Both arrow directions in that union mean precisely a reversible edge.
        groups = defaultdict(list)
        for a in all_dags(4):
            groups[((a | a.T).tobytes(), oracle_colliders(a))].append(a)
        self.assertEqual(sum(map(len, groups.values())), 543)
        for dags in groups.values():
            expected = np.logical_or.reduce(dags).astype(np.uint8)
            for a in dags:
                np.testing.assert_array_equal(dag_to_cpdag(a), expected)

    def test_cpdag_five_node_permutation_oracle(self):
        rng = np.random.default_rng(31)
        for _ in range(30):
            a = np.triu(rng.random((5, 5)) < 0.45, 1).astype(np.uint8)
            permutation = rng.permutation(5)
            a = a[np.ix_(permutation, permutation)]
            s = a | a.T
            expected = np.zeros_like(a)
            colliders = oracle_colliders(a)
            for order in itertools.permutations(range(5)):
                candidate = dag_for_order(s, order)
                if oracle_colliders(candidate) == colliders:
                    expected |= candidate
            np.testing.assert_array_equal(dag_to_cpdag(a), expected)

    def test_cpdag_and_targets_are_permutation_equivariant(self):
        a = np.zeros((5, 5), dtype=np.uint8)
        a[0, 2] = a[1, 2] = a[2, 3] = a[0, 4] = a[4, 3] = 1
        permutation = np.array([3, 0, 4, 1, 2])
        b = a[np.ix_(permutation, permutation)]
        np.testing.assert_array_equal(dag_to_cpdag(b),
                                      dag_to_cpdag(a)[np.ix_(permutation, permutation)])
        for left, right in zip(collider_targets(b), collider_targets(a)):
            np.testing.assert_array_equal(left, right[np.ix_(permutation, permutation, permutation)])

    def test_exact_cpdag_decoder_optimizes_collider_likelihood(self):
        rng = np.random.default_rng(9)
        for _ in range(8):
            upper = np.triu(rng.random((5, 5)) < 0.65, 1)
            s = (upper | upper.T).astype(np.uint8)
            probabilities = rng.uniform(0.03, 0.97, (5, 5, 5))
            result = decode_cpdag(s, probabilities, beam_width=None)
            expected_score = max(collider_score(dag_for_order(s, order), probabilities)
                                 for order in itertools.permutations(range(5)))
            self.assertAlmostEqual(result["score"], expected_score, places=10)
            np.testing.assert_array_equal(skeleton_from_dag(result["dag"]), s)
            np.testing.assert_array_equal(result["cpdag"], dag_to_cpdag(result["dag"]))

    def test_oracle_predictions_recover_cpdag(self):
        for a in all_dags(4):
            labels, _ = collider_targets(a)
            probabilities = labels * 0.98 + 0.01
            result = decode_cpdag(a | a.T, probabilities, beam_width=None)
            np.testing.assert_array_equal(result["cpdag"], dag_to_cpdag(a))
            self.assertEqual(result["collider_disagreements"], 0)

    def test_inconsistent_colliders_still_produce_valid_cpdag(self):
        # Adjacent colliders on a four-node path require contradictory arrows.
        s = np.zeros((4, 4), dtype=np.uint8)
        for i in range(3):
            s[i, i + 1] = s[i + 1, i] = 1
        c = np.full((4, 4, 4), 0.01)
        c[0, 1, 2] = c[2, 1, 0] = c[1, 2, 3] = c[3, 2, 1] = 0.99
        result = decode_cpdag(s, c, beam_width=None)
        self.assertEqual(result["rejected_colliders"], 1)
        self.assertTrue(oracle_is_dag(result["dag"]))
        np.testing.assert_array_equal(result["cpdag"], dag_to_cpdag(result["dag"]))
        # A chordless undirected cycle admits no acyclic orientation without a
        # collider, so low collider scores cannot all be honored simultaneously.
        s[0, 3] = s[3, 0] = 1
        result = decode_cpdag(s, np.full_like(c, 0.01), beam_width=None)
        self.assertGreaterEqual(result["introduced_colliders"], 1)

    def test_directed_decoder_handles_cycles_and_order_ties(self):
        s = np.ones((3, 3)) - np.eye(3)
        p = np.full((3, 3), 0.1)
        p[0, 1] = p[1, 2] = p[2, 0] = 0.9
        for beam in (1, 3, None):
            result = decode_dag(s, direction_probs=p, beam_width=beam)
            self.assertTrue(oracle_is_dag(result["dag"]))
            np.testing.assert_array_equal(skeleton_from_dag(result["dag"]), s)
        result = decode_dag(s, order_scores=[0, 0, 1])
        self.assertFalse(result["dag"][0, 1] or result["dag"][1, 0])
        self.assertEqual(result["omitted_tied_edges"], 1)
        self.assertTrue(result["dag"][0, 2] and result["dag"][1, 2])

    def test_dag_refinement_preserves_every_four_node_cpdag(self):
        rng = np.random.default_rng(8)
        seen = set()
        for a in all_dags(4):
            cpdag = dag_to_cpdag(a)
            if cpdag.tobytes() in seen:
                continue
            seen.add(cpdag.tobytes())
            s = a | a.T
            p = rng.uniform(0.01, 0.99, (4, 4))
            for kwargs in ({"direction_probs": p}, {"order_scores": rng.normal(size=4)}):
                result = decode_dag(s, cpdag=cpdag, **kwargs)
                np.testing.assert_array_equal(dag_to_cpdag(result["dag"]), cpdag)
                compelled = cpdag.astype(bool) & ~cpdag.T.astype(bool)
                self.assertTrue(np.all(result["dag"][compelled] == 1))
                self.assertEqual(oracle_colliders(result["dag"]), oracle_colliders(a))

    def test_refinement_resolves_rank_conflicts_with_compelled_arrows(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        a[0, 2] = a[1, 2] = a[2, 3] = 1
        cpdag = dag_to_cpdag(a)
        # These order scores request the reverse of every compelled arrow.
        result = decode_dag(a | a.T, order_scores=[3, 4, 2, 1], cpdag=cpdag)
        np.testing.assert_array_equal(result["dag"], a)
        # The unshielded chain must not acquire a new collider in refinement.
        chain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        result = decode_dag(chain, cpdag=chain, order_scores=[0, 2, 1])
        self.assertEqual(oracle_colliders(result["dag"]), ())

    def test_direction_head_affects_refinement_with_fixed_order_scores(self):
        chain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        forward = np.full((3, 3), 0.5)
        forward[0, 1] = forward[1, 2] = 0.9
        forward[1, 0] = forward[2, 1] = 0.1
        # Fixed rank scores strongly prefer the reverse ordering. Directional
        # evidence must still influence the result when both heads are passed.
        ranks = np.array([100., 0., -100.])
        a = decode_dag(chain, cpdag=chain, direction_probs=forward,
                       order_scores=ranks)["dag"]
        b = decode_dag(chain, cpdag=chain, direction_probs=forward.T,
                       order_scores=ranks)["dag"]
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(a, expected)
        np.testing.assert_array_equal(b, expected.T)
        np.testing.assert_array_equal(dag_to_cpdag(a), chain)
        np.testing.assert_array_equal(dag_to_cpdag(b), chain)
        self.assertEqual(oracle_colliders(a), ())
        self.assertEqual(oracle_colliders(b), ())

    def test_decoder_validation_and_no_input_mutation(self):
        s = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        c = np.full((3, 3, 3), 0.1)
        s_copy, c_copy = s.copy(), c.copy()
        decode_cpdag(s, c)
        np.testing.assert_array_equal(s, s_copy)
        np.testing.assert_array_equal(c, c_copy)
        for kwargs in ({"beam_width": 0}, {"threshold": -1}, {"collider_threshold": 2}):
            with self.assertRaises(ValueError):
                decode_cpdag(s, c, **kwargs)
        with self.assertRaises(ValueError):
            decode_cpdag(s, np.full_like(c, np.nan))
        with self.assertRaises(ValueError):
            decode_dag(s)
        with self.assertRaises(ValueError):
            decode_dag(s, order_scores=[0, 1, np.inf])
        # A directed chain has reversible arrows and is not a completed PDAG.
        noncompleted = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        with self.assertRaises(ValueError):
            decode_dag(s, order_scores=[0, 1, 2], cpdag=noncompleted)
        with self.assertRaises(ValueError):
            decode_dag(s, order_scores=[0, 1, 2], cpdag=np.zeros((3, 3)))

    def test_empty_and_single_node_graphs(self):
        for d in (0, 1):
            s = np.zeros((d, d))
            result = decode_cpdag(s, np.zeros((d, d, d)), beam_width=None)
            self.assertEqual(result["cpdag"].shape, (d, d))
            self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
