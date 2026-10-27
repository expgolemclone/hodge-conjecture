"""Exact Hodge-number calculations for smooth complex projective hypersurfaces.

OPEN TARGET: for every smooth projective X/C and codimension p, the cycle map
CH^p(X) tensor Q -> H^(2p)(X,Q) intersect H^(p,p)(X,C) is surjective.
Official statement by Deligne:
https://www.claymath.org/wp-content/uploads/2022/06/hodge.pdf

This module only computes dimensions over C. In particular, h^(p,p) is NOT
the dimension over Q of rational Hodge classes. No algebraic cycles, cycle
class map, or proof of the Hodge conjecture are supplied.

For a smooth degree-d hypersurface X in P^(n+1), Griffiths' Jacobian-ring
description gives h_prim^(n-q,q) as the coefficient of t^((q+1)d-n-2) in
(1+t+...+t^(d-2))^(n+2). The partial derivatives form a regular sequence.
Reference: Griffiths, On the periods of certain rational integrals, I-II
(1969). Smoothness is a hypothesis, not something inferred from n and d.

Example: python -m research.hypersurfaces 4 3
Checks: python -B -m unittest discover -s checks -v
"""

from math import comb


def primitive_middle(dimension: int, degree: int) -> tuple[int, ...]:
    """Return h_prim^(n-q,q), indexed by q, under the smoothness hypothesis."""
    if type(dimension) is not int or type(degree) is not int:
        raise TypeError("Dimension and degree must be integers.")
    if dimension < 0 or degree < 1:
        raise ValueError("Dimension must be nonnegative and degree positive.")
    if degree == 1:
        return (0,) * (dimension + 1)

    variables = dimension + 2
    result = []
    for q in range(dimension + 1):
        exponent = (q + 1) * degree - variables
        # Expand (1-t^(d-1))^(n+2) / (1-t)^(n+2).
        # Negative exponents give an empty sum.
        coefficient = sum(
            (-1) ** j * comb(variables, j)
            * comb(exponent - j * (degree - 1) + variables - 1, variables - 1)
            for j in range(min(variables, exponent // (degree - 1)) + 1)
        )
        result.append(coefficient)
    return tuple(result)


def hodge_numbers(dimension: int, degree: int) -> dict[tuple[int, int], int]:
    """Return nonzero h^(p,q), adding the ambient projective-space classes."""
    primitive = primitive_middle(dimension, degree)
    numbers = {(p, p): 1 for p in range(dimension + 1)}
    for q, multiplicity in enumerate(primitive):
        if multiplicity:
            bidegree = (dimension - q, q)
            numbers[bidegree] = numbers.get(bidegree, 0) + multiplicity
    return numbers


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Hodge numbers of a smooth hypersurface.")
    parser.add_argument("dimension", type=int)
    parser.add_argument("degree", type=int)
    args = parser.parse_args()
    for (p, q), multiplicity in sorted(hodge_numbers(args.dimension, args.degree).items()):
        print(f"h^({p},{q}) = {multiplicity}")
    print("Complex dimensions only, not rational Hodge classes or a proof.")
