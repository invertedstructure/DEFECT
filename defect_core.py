"""Core logic for the F₂ suspension‑defect experiment.

This module collects all of the mathematics and linear algebra required
to construct simplicial complexes, compute coboundaries, perform
row‑reduction over GF(2), and to assemble and evaluate the repeated
suspension experiment.  It contains no user interface code and can be
imported from both a command‑line script and a Streamlit app.

Functions:

* `closure` – ensure a complex is closed under faces.
* `build_cycle_complex` – build a cycle graph C_n as a simplicial complex.
* `build_basis` – provide a deterministic basis for q‑simplices.
* `build_coboundary_matrix` – construct δ^q over GF(2).
* `gf2_rank` – compute rank of a matrix over GF(2).
* `gf2_is_in_column_space` – test membership of a vector in the column span.
* `lift_cochain_to_dict` – convert vector cochains to dicts.
* `cochain_dict_to_vector` – convert dict cochains to vectors.
* `one_sided_suspension_cochain` – lift a cochain under suspension.
* `suspend_complex` – perform simplicial suspension.
* `random_relabel_complex_and_cochain` – anti‑cheat relabelling.
* `run_experiment` – orchestrate the entire test for a range of cycle lengths
  and suspension depths, returning detailed results and diagnostics.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple, Set, Any, Optional, Iterable


Simplex = Tuple[Any, ...]
Complex = Set[Simplex]
CoChainDict = Dict[Simplex, int]


def closure(complex_simplices: Complex) -> Complex:
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
    result: Complex = set()
    for simplex in complex_simplices:
        verts = list(simplex)
        for r in range(1, len(verts) + 1):
            for face in itertools.combinations(verts, r):
                sorted_face = tuple(sorted(face, key=lambda x: str(x)))
                result.add(sorted_face)
    return result


def build_cycle_complex(n: int) -> Complex:
    """Construct the simplicial complex of the cycle graph C_n.

    The cycle has vertices labelled 0..n−1 and edges between i and (i+1) mod n.
    Since the cycle has no 2‑simplices, the closure simply consists of
    vertices and edges.

    Args:
        n: length of the cycle (integer ≥ 3).

    Returns:
        A set of simplices (tuples) representing the closed simplicial complex C_n.
    """
    simplices: Complex = set()
    # vertices
    for i in range(n):
        simplices.add((i,))
    # edges
    for i in range(n):
        j = (i + 1) % n
        edge = tuple(sorted((i, j), key=lambda x: str(x)))
        simplices.add(edge)
    # closure ensures all faces are present (redundant here but uniform)
    return closure(simplices)


def build_basis(complex_simplices: Complex, q: int) -> List[Simplex]:
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


def build_coboundary_matrix(complex_simplices: Complex, q: int) -> List[List[int]]:
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
    matrix: List[List[int]] = []
    for sigma in qp1_basis:
        row = [0] * n
        sigma_set = set(sigma)
        for j, tau in enumerate(q_basis):
            if set(tau).issubset(sigma_set):
                row[j] = 1
        matrix.append(row)
    return matrix


def gf2_rank(matrix: List[List[int]]) -> int:
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
    A = [row[:] for row in matrix]
    m = len(A)
    n = len(A[0])
    rank = 0
    for col in range(n):
        pivot_row = None
        for r in range(rank, m):
            if A[r][col] == 1:
                pivot_row = r
                break
        if pivot_row is not None:
            # swap pivot into position
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
            # eliminate other rows
            for r in range(m):
                if r != rank and A[r][col] == 1:
                    row_r = A[r]
                    row_p = A[rank]
                    for c in range(col, n):
                        row_r[c] ^= row_p[c]
            rank += 1
    return rank


def gf2_is_in_column_space(matrix: List[List[int]], vector: List[int]) -> bool:
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
    if not matrix:
        # no rows: the only element in the image is the zero vector
        return all(x == 0 for x in vector)
    augmented = []
    for i, row in enumerate(matrix):
        augmented.append(row[:] + [vector[i] % 2])
    rank_A = gf2_rank(matrix)
    rank_aug = gf2_rank(augmented)
    return rank_A == rank_aug


def lift_cochain_to_dict(complex_simplices: Complex, q: int, vector: List[int]) -> CoChainDict:
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
    cochain: CoChainDict = {}
    for i, simplex in enumerate(basis):
        val = vector[i] % 2
        if val:
            cochain[simplex] = 1
    return cochain


def cochain_dict_to_vector(complex_simplices: Complex, q: int, cochain_dict: CoChainDict) -> List[int]:
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


def one_sided_suspension_cochain(
    complex_simplices: Complex,
    cochain_dict: CoChainDict,
    q: int,
    new_a: Any,
    new_b: Any,
) -> CoChainDict:
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
    lifted: CoChainDict = {}
    for sigma, val in cochain_dict.items():
        if val % 2 == 1:
            new_simplex = tuple(sorted((new_a,) + sigma, key=lambda x: str(x)))
            lifted[new_simplex] = 1
    return lifted


def suspend_complex(
    complex_simplices: Complex, new_a: Any, new_b: Any
) -> Complex:
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
    new_complex: Complex = set()
    for sigma in complex_simplices:
        sorted_sigma = tuple(sorted(sigma, key=lambda x: str(x)))
        new_complex.add(sorted_sigma)
    for sigma in complex_simplices:
        sorted_sigma = tuple(sorted(sigma, key=lambda x: str(x)))
        cone_a = tuple(sorted((new_a,) + sorted_sigma, key=lambda x: str(x)))
        cone_b = tuple(sorted((new_b,) + sorted_sigma, key=lambda x: str(x)))
        new_complex.add(cone_a)
        new_complex.add(cone_b)
    return closure(new_complex)


def random_relabel_complex_and_cochain(
    complex_simplices: Complex,
    cochain_dict: CoChainDict,
) -> Tuple[Complex, CoChainDict]:
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
    verts = set()
    for sigma in complex_simplices:
        verts.update(sigma)
    mapping: Dict[Any, Any] = {}
    used: Set[Any] = set()
    for v in verts:
        while True:
            candidate = ('rand', random.randint(0, 10**9))
            if candidate not in used:
                used.add(candidate)
                mapping[v] = candidate
                break
    K_new: Complex = set()
    for sigma in complex_simplices:
        new_sigma = tuple(sorted([mapping[v] for v in sigma], key=lambda x: str(x)))
        K_new.add(new_sigma)
    u_new: CoChainDict = {}
    for sigma, val in cochain_dict.items():
        new_sigma = tuple(sorted([mapping[v] for v in sigma], key=lambda x: str(x)))
        u_new[new_sigma] = val % 2
    return K_new, u_new


def run_experiment(
    n_min: int,
    n_max: int,
    max_depth: int,
    do_random_relabel: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Run the suspension‑defect experiment for a range of cycle lengths.

    This function orchestrates the entire test: it builds each cycle graph
    C_n, constructs the all‑ones edge cochain, suspends both the complex
    and the cochain repeatedly, and determines whether the lifted cochain
    lies in the image of the coboundary at each stage.  If random
    relabelling is requested, an additional anti‑cheat check is performed
    for each case.  All intermediate results are returned along with
    summary statistics and detailed failure diagnostics.

    Args:
        n_min: minimum cycle length (inclusive).
        n_max: maximum cycle length (inclusive).
        max_depth: maximum suspension depth (number of suspensions).
        do_random_relabel: if True, perform a single random relabelling check
            for each (n, k) pair to ensure label independence.

    Returns:
        A tuple (results, summary, failures) where:
          - results is a list of dictionaries, one per (n, k) pair, containing
            cycle length, suspension depth, degree, seed triviality, lifted
            triviality, preservation flag, and expected class by parity.
          - summary is a dictionary with total, passes, failures, and parity
            breakdown counts.
          - failures is a list of dictionaries containing diagnostic
            information for each failure encountered.  If no failures,
            this list is empty.
    """
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for n in range(n_min, n_max + 1):
        K0 = build_cycle_complex(n)
        # seed cochain: all ones on edges
        q0 = 1
        basis_edges = build_basis(K0, q0)
        seed_vector = [1] * len(basis_edges)
        seed_cochain = lift_cochain_to_dict(K0, q0, seed_vector)
        # seed triviality: b_n in im δ^0
        delta0 = build_coboundary_matrix(K0, 0)
        seed_is_trivial = gf2_is_in_column_space(delta0, cochain_dict_to_vector(K0, q0, seed_cochain))
        expected_class = "trivial" if n % 2 == 0 else "nontrivial"
        K_current = K0
        u_current = seed_cochain
        q_current = q0
        for k in range(0, max_depth + 1):
            delta_prev = build_coboundary_matrix(K_current, q_current - 1)
            v_current = cochain_dict_to_vector(K_current, q_current, u_current)
            lifted_is_trivial = gf2_is_in_column_space(delta_prev, v_current)
            preserved = (lifted_is_trivial == seed_is_trivial)
            results.append({
                "cycle_n": n,
                "suspension_depth": k,
                "degree": q_current,
                "seed_trivial": seed_is_trivial,
                "lifted_trivial": lifted_is_trivial,
                "preserved": preserved,
                "expected_by_parity": expected_class,
            })
            # anti‑cheat random relabel
            if do_random_relabel:
                K_rand, u_rand = random_relabel_complex_and_cochain(K_current, u_current)
                delta_prev_rand = build_coboundary_matrix(K_rand, q_current - 1)
                v_rand = cochain_dict_to_vector(K_rand, q_current, u_rand)
                lifted_rand = gf2_is_in_column_space(delta_prev_rand, v_rand)
                if lifted_rand != lifted_is_trivial:
                    preserved = False
                    failure = {
                        "cycle_n": n,
                        "suspension_depth": k,
                        "degree": q_current,
                        "seed_trivial": seed_is_trivial,
                        "lifted_trivial": lifted_is_trivial,
                        "lifted_rand_trivial": lifted_rand,
                        "expected_by_parity": expected_class,
                        "anti_cheat_failure": True,
                    }
                    failures.append(failure)
            # record mismatch failure if expectation not met
            if not preserved:
                # compute diagnostic data
                m = len(delta_prev)
                ncols = len(delta_prev[0]) if delta_prev else 0
                rank_A = gf2_rank(delta_prev)
                augmented: List[List[int]] = []
                for i, row in enumerate(delta_prev):
                    augmented.append(row[:] + [v_current[i]])
                rank_aug = gf2_rank(augmented)
                support = ["+".join(str(v) for v in simplex) for simplex in u_current.keys()]
                failure = {
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
                    "cochain_support": support,
                }
                failures.append(failure)
            # prepare next level
            if k < max_depth:
                depth = k + 1
                new_a = ("susp", depth, "a")
                new_b = ("susp", depth, "b")
                K_next = suspend_complex(K_current, new_a, new_b)
                u_next = one_sided_suspension_cochain(K_current, u_current, q_current, new_a, new_b)
                K_current = K_next
                u_current = u_next
                q_current = q_current + 1
        # end k loop
    # build summary
    total_cases = len(results)
    passes = sum(1 for r in results if r["preserved"])
    failures_count = total_cases - passes
    even_cases = [r for r in results if r["cycle_n"] % 2 == 0]
    odd_cases = [r for r in results if r["cycle_n"] % 2 == 1]
    summary = {
        "total_cases": total_cases,
        "passes": passes,
        "failures": failures_count,
        "even_cases": len(even_cases),
        "even_failures": sum(1 for r in even_cases if not r["preserved"]),
        "odd_cases": len(odd_cases),
        "odd_failures": sum(1 for r in odd_cases if not r["preserved"]),
    }
    return results, summary, failures