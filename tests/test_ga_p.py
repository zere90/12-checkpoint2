"""Danial's GA-P checks. Run: python -m unittest tests.test_ga_p -v."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta import ga_problem as problem


class GAPTests(unittest.TestCase):
    def setUp(self):
        self.d = cp34_instance(12, "B", 2)
        self.rng = np.random.default_rng(73)
        self.original_rho = problem.RHO

    def tearDown(self):
        problem.RHO = self.original_rho

    def assert_valid(self, d, a):
        self.assertEqual(a.shape, (d["M"],))
        self.assertIn(a.dtype.kind, "iu")
        self.assertTrue(np.all((a >= 0) & (a < d["S"])))
        required = d["vm_required_zone"]
        self.assertTrue(np.all((required == -1) | (d["zone"][a] == required)))

    def test_checker_agreement_on_1000_random_individuals(self):
        feasible_count = 0
        for _ in range(1000):
            a = problem.random_individual(self.d, 2, self.rng)
            self.assert_valid(self.d, a)
            fitness, cost, feasible = problem.evaluate(self.d, 2, a)
            true_cost, problems = evaluate_B(self.d, a)
            self.assertEqual(cost, true_cost)
            self.assertEqual(feasible, not bool(problems))
            self.assertAlmostEqual(
                fitness, true_cost + problem.RHO * problem.capacity_violation(self.d, a))
            feasible_count += feasible
        # Ensure this sample exercised both capacity-feasible and overloaded states.
        self.assertGreater(feasible_count, 0)
        self.assertLess(feasible_count, 1000)

    def test_operators_on_1000_pairs_without_parent_changes(self):
        for _ in range(1000):
            p1 = problem.random_individual(self.d, 2, self.rng)
            p2 = problem.random_individual(self.d, 2, self.rng)
            saved1, saved2 = p1.copy(), p2.copy()
            child = problem.crossover(p1, p2, self.rng)
            self.assert_valid(self.d, child)
            self.assertTrue(np.all((child == p1) | (child == p2)))
            saved_child = child.copy()
            mutated = problem.mutate(self.d, 2, child, self.rng)
            self.assert_valid(self.d, mutated)
            self.assertIn(int(np.count_nonzero(mutated != child)), (1, 2))
            np.testing.assert_array_equal(p1, saved1)
            np.testing.assert_array_equal(p2, saved2)
            np.testing.assert_array_equal(child, saved_child)
            self.assertFalse(np.shares_memory(child, p1))
            self.assertFalse(np.shares_memory(mutated, child))

    @staticmethod
    def small_instance():
        # Both VMs on server 0: CPU=6 > 4, memory=10 > 8.
        # Relative overload = 2/4 + 2/8 = 0.75, computed by hand.
        return dict(story="B", variant=2, S=2, M=2,
                    cpu_cap=np.array([4, 4]), mem_cap=np.array([8, 8]),
                    cpu_req=np.array([3, 3]), mem_req=np.array([5, 5]),
                    zone=np.array([0, 0]), vm_required_zone=np.array([-1, -1]))

    def test_hand_penalty_and_true_cost_are_separate(self):
        d = self.small_instance()
        cost, problems = evaluate_B(d, [0, 0])
        self.assertTrue(problems)
        self.assertEqual(problem.capacity_violation(d, [0, 0]), 0.75)
        for rho in (1.0, 10.0, 100.0):
            problem.RHO = rho
            fitness, reported_cost, feasible = problem.evaluate(d, 2, [0, 0])
            self.assertEqual(fitness, cost + rho * 0.75)
            self.assertEqual(reported_cost, cost)
            self.assertFalse(feasible)
        fitness, cost, feasible = problem.evaluate(d, 2, [0, 1])
        self.assertEqual(fitness, cost)
        self.assertTrue(feasible)

    def test_one_actual_checker_call_per_evaluation(self):
        a = problem.random_individual(self.d, 2, self.rng)
        with patch.object(problem, "check", wraps=problem.check) as checker:
            problem.evaluate(self.d, 2, a)
            self.assertEqual(checker.call_count, 1)

    def test_overload_is_not_repaired_or_filtered(self):
        d = self.small_instance()
        # One allowed server per VM; no capacity-feasible assignment exists.
        d["zone"] = np.array([0, 1])
        d["vm_required_zone"] = np.array([0, 0])
        a = problem.random_individual(d, 2, self.rng)
        np.testing.assert_array_equal(a, [0, 0])
        child = problem.crossover(a, a, self.rng)
        mutated = problem.mutate(d, 2, child, self.rng)
        np.testing.assert_array_equal(mutated, [0, 0])
        self.assertFalse(problem.evaluate(d, 2, mutated)[2])

    def test_other_instances_and_rng_repeatability(self):
        for seed in (1, 8, 21, 56):
            d = cp34_instance(seed, "B", 2)
            r1, r2 = np.random.default_rng(45), np.random.default_rng(45)
            # Legacy global RNG state must neither be consulted nor changed.
            state_before = np.random.get_state()
            p1 = problem.random_individual(d, 2, r1)
            p2 = problem.random_individual(d, 2, r2)
            np.testing.assert_array_equal(p1, p2)
            for _ in range(5):
                p1 = problem.mutate(d, 2, problem.crossover(p1, p1, r1), r1)
                p2 = problem.mutate(d, 2, problem.crossover(p2, p2, r2), r2)
                np.testing.assert_array_equal(p1, p2)
                self.assert_valid(d, p1)
            after = np.random.get_state()
            self.assertEqual(state_before[0], after[0])
            np.testing.assert_array_equal(state_before[1], after[1])
            self.assertEqual(state_before[2:], after[2:])

    def test_wrong_zone_gets_checker_cost_and_infinite_fitness(self):
        d = self.small_instance()
        d["zone"] = np.array([0, 1])
        d["vm_required_zone"] = np.array([0, -1])
        fitness, cost, feasible = problem.evaluate(d, 2, [1, 0])
        self.assertTrue(np.isinf(fitness))
        self.assertEqual(cost, evaluate_B(d, [1, 0])[0])
        self.assertFalse(feasible)
        with self.assertRaises(ValueError):
            problem.mutate(d, 2, [1, 0], self.rng)

    def test_malformed_assignments_and_impossible_zones(self):
        d = self.small_instance()
        for a in ([0], [0, 2], [-1, 0], [0.5, 1.0], [[0, 1]]):
            with self.assertRaises(ValueError):
                problem.evaluate(d, 2, a)
        d["vm_required_zone"][0] = 99
        with self.assertRaises(ValueError):
            problem.random_individual(d, 2, self.rng)

    def test_bad_rho_and_wrong_variant(self):
        d = self.small_instance()
        for rho in (0, -1, float("nan"), float("inf")):
            problem.RHO = rho
            with self.assertRaises(ValueError):
                problem.evaluate(d, 2, [0, 1])
        with self.assertRaises(ValueError):
            problem.random_individual(d, 1, self.rng)


if __name__ == "__main__":
    unittest.main()
