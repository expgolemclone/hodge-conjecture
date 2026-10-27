"""Regression cases and independent topological consistency checks."""

import unittest

from research.hypersurfaces import hodge_numbers, primitive_middle


class HypersurfaceChecks(unittest.TestCase):
    def test_hyperplanes_are_projective_spaces(self):
        for n in range(7):
            with self.subTest(dimension=n):
                self.assertEqual(primitive_middle(n, 1), (0,) * (n + 1))
                self.assertEqual(hodge_numbers(n, 1), {(p, p): 1 for p in range(n + 1)})

    def test_zero_dimensional_hypersurface_is_degree_many_points(self):
        for degree in range(1, 10):
            self.assertEqual(hodge_numbers(0, degree), {(0, 0): degree})

    def test_plane_curve_genus(self):
        for degree in range(1, 12):
            genus = (degree - 1) * (degree - 2) // 2
            numbers = hodge_numbers(1, degree)
            self.assertEqual(primitive_middle(1, degree), (genus, genus))
            self.assertEqual(numbers.get((1, 0), 0), genus)
            self.assertEqual(numbers.get((0, 1), 0), genus)

    def test_quadric_surface(self):
        self.assertEqual(primitive_middle(2, 2), (0, 1, 0))
        self.assertEqual(hodge_numbers(2, 2), {(0, 0): 1, (1, 1): 2, (2, 2): 1})

    def test_quartic_k3(self):
        self.assertEqual(primitive_middle(2, 4), (1, 19, 1))
        self.assertEqual(
            hodge_numbers(2, 4),
            {(0, 0): 1, (0, 2): 1, (1, 1): 20, (2, 0): 1, (2, 2): 1},
        )

    def test_quintic_threefold(self):
        self.assertEqual(primitive_middle(3, 5), (1, 101, 101, 1))
        numbers = hodge_numbers(3, 5)
        self.assertEqual(numbers[(1, 1)], 1)
        self.assertEqual(numbers[(2, 1)], 101)
        self.assertEqual(numbers[(3, 0)], 1)

    def test_cubic_fourfold_keeps_primitive_and_ambient_classes_separate(self):
        self.assertEqual(primitive_middle(4, 3), (0, 1, 20, 1, 0))
        self.assertEqual(hodge_numbers(4, 3)[(2, 2)], 21)
        # Neither 20 nor 21 is asserted to be the rational Hodge-class rank.

    def test_symmetries_and_euler_characteristic(self):
        for n in range(7):
            for degree in range(1, 8):
                with self.subTest(dimension=n, degree=degree):
                    numbers = hodge_numbers(n, degree)
                    for (p, q), multiplicity in numbers.items():
                        self.assertIs(type(multiplicity), int)
                        self.assertGreater(multiplicity, 0)
                        self.assertEqual(numbers.get((q, p), 0), multiplicity)
                        self.assertEqual(numbers.get((n - p, n - q), 0), multiplicity)
                    # Chern-class computation, independent of the Jacobian-ring sum.
                    numerator = (1 - degree) ** (n + 2) - 1
                    self.assertEqual(numerator % degree, 0)
                    euler = n + 2 + numerator // degree
                    self.assertEqual(
                        sum((-1) ** (p + q) * h for (p, q), h in numbers.items()),
                        euler,
                    )

    def test_invalid_parameters(self):
        for n, degree in [(-1, 3), (2, 0), (2, -1)]:
            with self.assertRaises(ValueError):
                hodge_numbers(n, degree)
        for n, degree in [(2.0, 3), (2, 3.0), (True, 3), (2, False), ("2", 3)]:
            with self.assertRaises(TypeError):
                hodge_numbers(n, degree)
