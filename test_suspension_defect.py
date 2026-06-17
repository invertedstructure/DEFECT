#!/usr/bin/env python3
"""
test_suspension_defect.py

This script implements a finite (F_2)-cohomology experiment to probe the
behaviour of a simple 1‑dimensional parity defect under repeated simplicial
suspension.  The seed defect lives on a cycle graph C_n as the all‑ones edge
cochain.  A one‑sided suspension operation is applied repeatedly to both
the complex and the cochain, and at each stage the script tests whether the
lifted defect remains trivial or nontrivial in cohomology.  The test is
completely agnostic about the parity of n beyond the initial seed: it uses
only linear algebra over GF(2) to determine coboundaries.  Random vertex
relabelings are used as an anti‑cheat check to ensure no hidden metadata
influences the result.

The output consists of JSON lines, one per (n, k) pair, describing the
observed behaviour together with the expected behaviour determined by the
parity of the cycle length.  A summary is printed at the end.  If any
expected preservation fails, additional debugging information is printed for
inspection and the script stops early.
"""

import itertools
import random
import json
import sys


def closure(complex_simplices):
    """Return the simplicial closure of a set of simplices.

    Given a set of simplices (each represented as a tuple of vertices), this
    function adds every non‑empty face of each simplex to ensure the result is
    closed under taking faces.  Vertices within a simplex and simplices
    themselves are stored as tuples sorted by the string representation of
    their vertices for determinism.

    Args:
        complex_simplices: set of tuples, each tuple listing the vertices of a simplex.

    Returns:
        A set of tuples representing the closure of the input complex.
    """
    result = set()
    for simplex in complex_simplices:
        verts = list(simplex)
        # add all non‑empty subsets (faces)
        for r in range(1, len(verts) + 1):
            for face in itertools.combinations(verts, r):
                sorted_face = tuple(sorted(face, key=lambda x: str(x)))
                result.add(sorted_face)
    return result


def build_cycle_complex(n):
    """Construct the simplicial complex of the cycle graph C_n.

    The cycle has vertices labelled 0..n−1 and edges between i and (i+1) mod n.
    Since the cycle has no 2‑simplices, the closure simply consists of
    vertices and edges.

    Args:
        n: length of the cycle (integer ≥ 3).

    Returns:
        A set of simplices (tuples) representing the closed simplicial complex C_n.
    """
    simplices = set()
    # vertices
    for i in range(n):
        simplices.add((i,))
    # edges
    for i in range(n):
        j = (i + 1) % n
        edge = tuple(sorted((i, j), key=lambda x: str(x)))
        simplices.add(edge)
    # closure is actually redundant here but applied for uniformity
    return closure(simplices)


def build_basis(complex_simplices, q):
    """Return a deterministic basis of q‑simplices of a complex.

    The basis is the list of all simplices in the complex of dimension q
    (i.e. cardinality q+1), sorted lexicographically by the string
    representation of their vertices.  The sorting ensures that different
    realisations of isomorphic complexes produce identical bases.

    Args:
        complex_simplices: set of tuples representing simplices.
        q: non‑negative integer specifying the desired simplex dimension.

    Returns:
        A list of q‑simplices (tuples) sorted deterministically.
    """
    simplices_q = [s for s in complex_simplices if len(s) == q + 1]
    return sorted(simplices_q, key=lambda s: [str(v) for v in s])


def build_coboundary_matrix(complex_simplices, q):
    """Construct the coboundary matrix δ^q for a simplicial complex.

    Over F_2 the orientation signs drop out, so an entry is 1 exactly when
    the q‑simplex is a face of the (q+1)‑simplex.

    Args:
        complex_simplices: set of simplices (tuples).
        q: integer ≥ 0 indicating which coboundary map δ^q to build.

    Returns:
        A list of rows, each row a list of 0/1 integers.  The matrix has
        dimension (# of (q+1)‑simplices) × (# of q‑simplices).  Rows
        correspond to (q+1)‑simplices, columns to q‑simplices.
    """
    q_basis = build_basis(complex_simplices, q)
    qp1_basis = build_basis(complex_simplices, q + 1)
    m = len(qp1_basis)
    n = len(q_basis)
    # build row for each (q+1)-simplex
    matrix = []
    for sigma in qp1_basis:
        row = [0] * n
        sigma_set = set(sigma)
        for j, tau in enumerate(q_basis):
            # tau is a face of sigma if its vertex set is subset of sigma
            if set(tau).issubset(sigma_set):
                row[j] = 1
        matrix.append(row)
    return matrix


def gf2_rank(matrix):
    """Compute the rank of a matrix over GF(2) via Gaussian elimination.

    The input matrix is a list of rows of equal length.  The algorithm
    performs row operations (swaps and additions) to produce a row‑echelon
    form, counting the number of pivot columns.  A copy of the input is
    made so the original matrix is not modified.

    Args:
        matrix: list of lists, each inner list representing a row of bits.

    Returns:
        An integer giving the rank of the matrix over GF(2).
    """
    if not matrix:
        return 0
    # make a deep copy so we don't mutate the input
    A = [row[:] for row in matrix]
    m = len(A)
    n = len(A[0])
    rank = 0
    # iterate over columns
    for col in range(n):
        # find a pivot row at or below the current rank with a 1 in this column
        pivot_row = None
        for r in range(rank, m):
            if A[r][col] == 1:
                pivot_row = r
                break
        if pivot_row is not None:
            # move pivot row into position 'rank'
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
            # eliminate 1s in this column for all other rows
            for r in range(m):
                if r != rank and A[r][col] == 1:
                    # row_r = row_r + row_rank (mod 2)
                    row_r = A[r]
                    row_p = A[rank]
                    # only operate on columns from 'col' onwards since earlier
                    # columns are already in echelon form
                    for c in range(col, n):
                        row_r[c] ^= row_p[c]
            # increment rank and move to next column
            rank += 1
    return rank


def gf2_is_in_column_space(matrix, vector):
    """Test whether a vector is in the column space of a matrix over GF(2).

    To determine if a vector v lies in the span of the columns of A, we
    compute the rank of A and the rank of the augmented matrix [A|v].  The
    ranks are equal iff v is in the column space of A.  The function makes
    copies of its inputs so no mutation of the original data structures
    occurs.

    Args:
        matrix: list of rows (list of ints 0/1) representing an m×n matrix A.
        vector: list of length m of ints 0/1 representing v ∈ F_2^m.

    Returns:
        True if v ∈ im(A), False otherwise.
    """
    # When there are no rows, the only vector in the image is the zero vector.
    if not matrix:
        return all(x == 0 for x in vector)
    # build augmented matrix by appending v as an extra column
    aug = []
    for i, row in enumerate(matrix):
        new_row = row[:] + [vector[i] % 2]
        aug.append(new_row)
    rank_A = gf2_rank(matrix)
    rank_aug = gf2_rank(aug)
    return rank_A == rank_aug


def lift_cochain_to_dict(complex_simplices, q, vector):
    """Convert a cochain vector to a dictionary keyed by simplices.

    Args:
        complex_simplices: set of simplices for the complex.
        q: degree of the cochain (dimension of simplices).
        vector: list of integers (0/1) representing the cochain in the
                canonical basis order returned by build_basis.

    Returns:
        A dictionary mapping each q‑simplex (tuple) to its coefficient (0 or 1).
    """
    basis = build_basis(complex_simplices, q)
    cochain = {}
    for i, simplex in enumerate(basis):
        val = vector[i] % 2
        if val:
            cochain[simplex] = 1
    return cochain


def cochain_dict_to_vector(complex_simplices, q, cochain_dict):
    """Convert a dictionary cochain to a vector in canonical basis order.

    Args:
        complex_simplices: set of simplices for the complex.
        q: degree of the cochain (dimension of simplices).
        cochain_dict: dictionary mapping q‑simplices to coefficients (0 or 1).

    Returns:
        A list of integers representing the cochain in the canonical basis.
    """
    basis = build_basis(complex_simplices, q)
    return [cochain_dict.get(simplex, 0) % 2 for simplex in basis]


def one_sided_suspension_cochain(complex_simplices, cochain_dict, q, new_a, new_b):
    """Perform the one‑sided suspension of a cochain.

    For a q‑cochain u on K, the lifted (q+1)‑cochain S(u) on ΣK is defined
    by S(u)(a*σ) = u(σ), S(u)(b*σ) = 0 for every q‑simplex σ of K, and
    S(u) zero on all other (q+1)‑simplices.  Only those cones over
    simplices where u takes value 1 contribute to the lifted cochain.

    Args:
        complex_simplices: set of simplices of K (unused here but kept for signature symmetry).
        cochain_dict: dictionary mapping q‑simplices to 0/1 values.
        q: degree of the cochain (used for clarity).
        new_a: label for the new suspension vertex on one side.
        new_b: label for the new suspension vertex on the other side.

    Returns:
        A dictionary mapping (q+1)‑simplices in ΣK to 0/1 values.
    """
    lifted = {}
    for sigma, val in cochain_dict.items():
        if val % 2 == 1:
            # the new simplex is (new_a,) + sigma, sorted deterministically
            new_simplex = tuple(sorted((new_a,) + sigma, key=lambda x: str(x)))
            lifted[new_simplex] = 1
    return lifted


def suspend_complex(complex_simplices, new_a, new_b):
    """Construct the simplicial suspension ΣK of a complex K.

    Two new vertices (new_a, new_b) are introduced, and for each simplex σ in
    K the cones (new_a, σ) and (new_b, σ) are added.  The result is then
    closed under taking faces.

    Args:
        complex_simplices: set of simplices representing K.
        new_a: label for one suspension vertex.
        new_b: label for the other suspension vertex.

    Returns:
        A set of simplices representing ΣK.
    """
    new_complex = set()
    # include all original simplices
    for sigma in complex_simplices:
        sorted_sigma = tuple(sorted(sigma, key=lambda x: str(x)))
        new_complex.add(sorted_sigma)
    # add cones over each simplex
    for sigma in complex_simplices:
        sorted_sigma = tuple(sorted(sigma, key=lambda x: str(x)))
        cone_a = tuple(sorted((new_a,) + sorted_sigma, key=lambda x: str(x)))
        cone_b = tuple(sorted((new_b,) + sorted_sigma, key=lambda x: str(x)))
        new_complex.add(cone_a)
        new_complex.add(cone_b)
    # take closure under faces
    return closure(new_complex)


def random_relabel_complex_and_cochain(complex_simplices, cochain_dict):
    """Produce a randomly relabelled copy of a complex and its cochain.

    All vertices in the complex are assigned new unique random labels.  The
    structure of the complex (simplices) and the cochain support are preserved
    under this relabelling.  This function acts as an anti‑cheat check to
    ensure that no metadata about the original labels influences the
    cohomology computation.

    Args:
        complex_simplices: set of simplices.
        cochain_dict: dictionary mapping simplices to 0/1 values.

    Returns:
        (K_new, u_new) where K_new is the relabelled complex and u_new is the
        relabelled cochain dictionary.
    """
    # gather all vertices
    verts = set()
    for sigma in complex_simplices:
        verts.update(sigma)
    # create random bijection to new labels
    mapping = {}
    used = set()
    for v in verts:
        # generate a unique random label as a tuple
        while True:
            candidate = ('rand', random.randint(0, 10**9))
            if candidate not in used:
                used.add(candidate)
                mapping[v] = candidate
                break
    # relabel complex
    K_new = set()
    for sigma in complex_simplices:
        new_sigma = tuple(sorted([mapping[v] for v in sigma], key=lambda x: str(x)))
        K_new.add(new_sigma)
    # relabel cochain
    u_new = {}
    for sigma, val in cochain_dict.items():
        new_sigma = tuple(sorted([mapping[v] for v in sigma], key=lambda x: str(x)))
        u_new[new_sigma] = val % 2
    return K_new, u_new


def main():
    results = []
    first_failure_info = None
    # iterate over cycle lengths n from 3 to 10
    for n in range(3, 11):
        # build the cycle complex C_n
        K0 = build_cycle_complex(n)
        # build the seed 1‑cochain b_n: all ones on edges (q=1)
        q = 1
        basis_edges = build_basis(K0, q)
        seed_vector = [1] * len(basis_edges)
        seed_cochain = lift_cochain_to_dict(K0, q, seed_vector)
        # compute seed triviality: whether b_n ∈ im δ^0
        delta0 = build_coboundary_matrix(K0, 0)
        seed_is_trivial = gf2_is_in_column_space(delta0, cochain_dict_to_vector(K0, q, seed_cochain))
        # expected behaviour from parity
        expected_class = "trivial" if n % 2 == 0 else "nontrivial"
        # set current complex and cochain for suspension iterations
        K_current = K0
        u_current = seed_cochain
        q_current = q
        # iterate over suspension depths k = 0..4
        for k in range(0, 5):
            # compute triviality of the current cochain u_current in K_current
            # trivial means u_current ∈ im δ^{q_current−1}
            delta_prev = build_coboundary_matrix(K_current, q_current - 1)
            v_current = cochain_dict_to_vector(K_current, q_current, u_current)
            lifted_is_trivial = gf2_is_in_column_space(delta_prev, v_current)
            # preservation: does this match the seed parity status?
            preserved = (lifted_is_trivial == seed_is_trivial)
            # append result row
            results.append({
                "cycle_n": n,
                "suspension_depth": k,
                "degree": q_current,
                "seed_trivial": seed_is_trivial,
                "lifted_trivial": lifted_is_trivial,
                "preserved": preserved,
                "expected_by_parity": expected_class
            })
            # anti‑cheat: random relabeling should not change triviality
            K_rand, u_rand = random_relabel_complex_and_cochain(K_current, u_current)
            delta_prev_rand = build_coboundary_matrix(K_rand, q_current - 1)
            v_rand = cochain_dict_to_vector(K_rand, q_current, u_rand)
            lifted_rand = gf2_is_in_column_space(delta_prev_rand, v_rand)
            if lifted_rand != lifted_is_trivial:
                preserved = False
                # record failure details if not already recorded
                if first_failure_info is None:
                    first_failure_info = {
                        "cycle_n": n,
                        "suspension_depth": k,
                        "degree": q_current,
                        "seed_trivial": seed_is_trivial,
                        "lifted_trivial": lifted_is_trivial,
                        "lifted_rand_trivial": lifted_rand,
                        "expected_by_parity": expected_class,
                        "anti_cheat_failure": True
                    }
            # detect mismatches with expected behaviour and record first failure
            if preserved is False and first_failure_info is None:
                # compute diagnostic data
                m = len(delta_prev)  # number of (q_current)-simplices
                ncols = len(delta_prev[0]) if delta_prev else 0  # number of (q_current-1)-simplices
                rank_A = gf2_rank(delta_prev)
                # build augmented for diagnostic
                augmented = []
                for i, row in enumerate(delta_prev):
                    augmented.append(row[:] + [v_current[i]])
                rank_aug = gf2_rank(augmented)
                # list support of cochain
                support = ["+".join(str(v) for v in simplex) for simplex in u_current.keys()]
                first_failure_info = {
                    "cycle_n": n,
                    "suspension_depth": k,
                    "degree": q_current,
                    "seed_trivial": seed_is_trivial,
                    "lifted_trivial": lifted_is_trivial,
                    "expected_by_parity": expected_class,
                    "matrix_rows": m,
                    "matrix_cols": ncols,
                    "rank_delta": rank_A,
                    "rank_aug": rank_aug,
                    "cochain_support": support
                }
            # prepare for the next suspension level
            if k < 4:
                depth = k + 1
                # generate new vertex labels for suspension at this depth
                new_a = ("susp", depth, "a")
                new_b = ("susp", depth, "b")
                # build the suspended complex and lift the cochain
                K_next = suspend_complex(K_current, new_a, new_b)
                u_next = one_sided_suspension_cochain(K_current, u_current, q_current, new_a, new_b)
                # update current data
                K_current = K_next
                u_current = u_next
                q_current = q_current + 1
        # end of k loop
    # print all result rows
    for row in results:
        print(json.dumps(row))
    # print summary
    total_cases = len(results)
    passes = sum(1 for r in results if r["preserved"])
    fails = total_cases - passes
    print("summary:")
    print(f"total cases: {total_cases}")
    print(f"passes: {passes}")
    print(f"failures: {fails}")
    # print first failure if any
    if first_failure_info:
        print("first failure:")
        print(json.dumps(first_failure_info, indent=2))
    # group by parity for summary
    even_cases = [r for r in results if r["cycle_n"] % 2 == 0]
    odd_cases = [r for r in results if r["cycle_n"] % 2 == 1]
    print("even cycles: total cases:", len(even_cases), "failures:", sum(1 for r in even_cases if not r["preserved"]))
    print("odd cycles: total cases:", len(odd_cases), "failures:", sum(1 for r in odd_cases if not r["preserved"]))


if __name__ == "__main__":
    main()