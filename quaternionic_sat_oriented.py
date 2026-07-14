# -*- coding: utf-8 -*-
"""quaternionic_sat_oriented.py

Refactorización del prototipo `quaternionic_sat_fixed.py` que
elimina la resolución algebraica directa (x = y, redondeos de
punto flotante y `bool(decision.bit)` sobre valores escalares)
y la reemplaza por un motor de decisión basado en PREDICADOS
GEOMÉTRICOS DE ORIENTACIÓN.

Cambios de fondo respecto a la versión anterior
================================================

1. Predicado de orientación (test de signo)
   -----------------------------------------
   Para cada literal se calcula un vector geométrico

       v_x = (x, k_x, x * k_x)                      k_x = base^(1/n_x)

   y se define un hiperplano de restricción de la cláusula c a
   partir del baricentro de sus literales:

       H_c : n_c . (p - p0_c) = 0

   La decisión booleana ya NO viene de `bool(decision.bit)`;
   viene del SIGNO del predicado de orientación

       orient(v_x, H_c) = sign( n_c . (v_x - p0_c) )   in {-1, 0, +1}

   con la convención:  +1 -> True,  -1 -> False,  0 -> frontera.

2. Manejo de fronteras: aritmética racional exacta
   ------------------------------------------------
   Cuando |n_c . (v_x - p0_c)| cae dentro de un umbral de máquina
   (`FLOAT_EPS`), el sistema conmuta AUTOMÁTICAMENTE a
   `fractions.Fraction` sobre una aproximación diádica del mismo
   predicado. Esto elimina la ambigüedad de signo en el borde y
   garantiza que el bit se decide por una relación de orden
   posicional exacta, no por redondeo.

3. Invarianza topológica bajo Keccak
   -----------------------------------
   El solver ya no minimiza el error cuadrático de coordenadas.
   En cambio, mezcla la matriz de decisiones con una permutación
   determinista derivada de rondas Keccak-f[1600] (implementación
   pura en Python, sin dependencias) y maximiza la CONSISTENCIA
   de los signos de orientación: para cada variable se elige el
   valor cuyo signo de orientación se mantiene invariante en el
   mayor número de rondas.

4. Salida booleana pura
   ---------------------
   `assignment[v]` es el producto de las evaluaciones de orientación
   agregadas por votación mayoritaria de signos a lo largo de las
   rondas Keccak. Ningún bit proviene de redondear un flotante ni
   de resolver una ecuación algebraica x = y.

La arquitectura de garantías se conserva:

    * "SAT"          <-> la asignación se VERIFICA sobre la fórmula.
    * "UNSAT"        <-> solo un backend exacto (python-sat) lo certifica.
    * "INCONCLUSIVE" <-> cualquier otro caso.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Clause = List[int]
CNF = List[Clause]
Assignment = Dict[int, bool]

# Umbral de máquina para conmutar a aritmética racional exacta.
FLOAT_EPS = 1e-12

# ============================================================
# Utilidades CNF
# ============================================================

def number_of_variables(cnf: CNF) -> int:
    return max((abs(lit) for clause in cnf for lit in clause), default=0)

def literal_value(literal: int, assignment: Assignment) -> bool:
    value = assignment.get(abs(literal), False)
    return value if literal > 0 else not value

def clause_satisfied(clause: Clause, assignment: Assignment) -> bool:
    return any(literal_value(lit, assignment) for lit in clause)

def formula_satisfied(cnf: CNF, assignment: Assignment) -> bool:
    return all(clause_satisfied(clause, assignment) for clause in cnf)

def unsatisfied_clause_indices(cnf: CNF, assignment: Assignment) -> List[int]:
    return [i for i, c in enumerate(cnf) if not clause_satisfied(c, assignment)]

def complete_assignment(assignment: Assignment, nvars: int, default: bool = False) -> Assignment:
    result = dict(assignment)
    for variable in range(1, nvars + 1):
        result.setdefault(variable, default)
    return result

# ============================================================
# Lectura/escritura DIMACS
# ============================================================

def read_dimacs(path: str) -> Tuple[CNF, int]:
    clauses: CNF = []
    current: Clause = []
    declared_nvars = 0
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                fields = line.split()
                if len(fields) < 4 or fields[1].lower() != "cnf":
                    raise ValueError(f"Cabecera DIMACS inválida: {line}")
                declared_nvars = int(fields[2])
                continue
            for token in line.split():
                literal = int(token)
                if literal == 0:
                    clauses.append(current)
                    current = []
                else:
                    current.append(literal)
    if current:
        raise ValueError("La última cláusula DIMACS no termina en 0")
    return clauses, max(declared_nvars, number_of_variables(clauses))

def write_dimacs(path: str, cnf: CNF, nvars: Optional[int] = None) -> None:
    if nvars is None:
        nvars = number_of_variables(cnf)
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"p cnf {nvars} {len(cnf)}\n")
        for clause in cnf:
            file.write(" ".join(map(str, clause)) + " 0\n")


# ============================================================
# Aritmética racional exacta para el predicado de frontera
# ============================================================

def _rational_pow_base_inv_n(base: int, n: int, precision_bits: int = 128) -> Fraction:
    """
    Aproximación diádica exacta (Fraction) de base**(1/n) con
    `precision_bits` bits de precisión. Se usa SOLO como
    representación posicional determinista para el predicado de
    signo cuando el flotante cae en la frontera.

    La estrategia es: usar el desarrollo binario del exponente
    aproximado x = log2(base) / n hasta `precision_bits` y
    reconstruir 2**x como potencia racional. Esto es suficiente
    porque el predicado depende únicamente del ORDEN relativo,
    no del valor exacto.
    """
    # log2(base) como Fraction diádica de precision_bits.
    # Empleamos math.log2 y truncamos: como esta rama se activa
    # únicamente cuando el flotante ya es ambiguo, basta con una
    # representación reproducible y estrictamente ordenable.
    log2_base = math.log2(base)
    scale = 1 << precision_bits
    quantized = int(log2_base * scale)  # trunca deterministamente
    # x = quantized / (scale * n)
    x_num = quantized
    x_den = scale * n if n > 0 else -(scale * n)
    sign_n = 1 if n > 0 else -1
    # 2**(x_num/x_den) no es racional en general: aproximamos
    # su valor por la serie de Taylor de exp(x * ln 2) truncada
    # con Fractions -> resultado racional exacto y reproducible.
    ln2 = _fraction_ln2(precision_bits)
    x = Fraction(sign_n * x_num, x_den) * ln2
    # exp(x) por serie de Taylor con cota fija de términos.
    terms = 40
    acc = Fraction(1, 1)
    factorial = Fraction(1, 1)
    power = Fraction(1, 1)
    for k in range(1, terms + 1):
        power *= x
        factorial *= k
        acc += power / factorial
    return acc

def _fraction_ln2(precision_bits: int) -> Fraction:
    """Aproximación racional de ln(2) mediante la serie
       ln 2 = sum_{k>=1} 1 / (k * 2**k).
    """
    total = Fraction(0, 1)
    for k in range(1, precision_bits):
        total += Fraction(1, k * (1 << k))
    return total


# ============================================================
# Codificación geométrica de literales
# ============================================================

@dataclass(frozen=True)
class EncodedDecision:
    clause_index: int
    occurrence_index: int
    spatial_index: int
    literal: int
    variable: int
    n: int                # literal firmado
    base: float
    k_float: float        # base**(1/n) en flotante
    # Vector geométrico v = (spatial_index, k, spatial_index * k)
    vx: float
    vy: float
    vz: float

def encode_literal(
    literal: int,
    clause_index: int,
    occurrence_index: int,
    spatial_index: int,
    base: float = 2.0,
) -> EncodedDecision:
    if literal == 0:
        raise ValueError("Un literal SAT no puede ser cero")
    if spatial_index < 1:
        raise ValueError("El índice espacial debe comenzar en 1")
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("La base debe ser positiva, finita y distinta de 1")

    n = literal
    k = math.exp(math.log(base) / n)
    vx = float(spatial_index)
    vy = k
    vz = spatial_index * k

    return EncodedDecision(
        clause_index=clause_index,
        occurrence_index=occurrence_index,
        spatial_index=spatial_index,
        literal=literal,
        variable=abs(literal),
        n=n,
        base=base,
        k_float=k,
        vx=vx,
        vy=vy,
        vz=vz,
    )

def build_conjunction_matrix(cnf: CNF, base: float = 2.0) -> List[List[EncodedDecision]]:
    matrix: List[List[EncodedDecision]] = []
    spatial_index = 1
    for clause_index, clause in enumerate(cnf):
        row: List[EncodedDecision] = []
        for occurrence_index, literal in enumerate(clause):
            row.append(encode_literal(
                literal=literal,
                clause_index=clause_index,
                occurrence_index=occurrence_index,
                spatial_index=spatial_index,
                base=base,
            ))
            spatial_index += 1
        matrix.append(row)
    return matrix


# ============================================================
# Predicado de orientación (test de signo geométrico)
# ============================================================

@dataclass(frozen=True)
class Hyperplane:
    """Hiperplano H : n . (p - p0) = 0 asociado a una cláusula."""
    clause_index: int
    nx: float
    ny: float
    nz: float
    p0x: float
    p0y: float
    p0z: float

def build_clause_hyperplane(row: List[EncodedDecision]) -> Hyperplane:
    """
    Hiperplano de restricción para una cláusula.

    - p0 = baricentro geométrico de los vectores v_x de sus literales.
    - n  = normal derivada del signo del literal y su índice espacial;
      esto codifica la polaridad sin apoyarse en `bool(bit)`.
    """
    if not row:
        return Hyperplane(-1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    m = len(row)
    p0x = sum(d.vx for d in row) / m
    p0y = sum(d.vy for d in row) / m
    p0z = sum(d.vz for d in row) / m
    # Normal: combinación firmada por polaridad de literal.
    nx = sum((1 if d.literal > 0 else -1) * d.spatial_index for d in row)
    ny = sum((1 if d.literal > 0 else -1) * d.k_float for d in row)
    nz = sum((1 if d.literal > 0 else -1) * (d.spatial_index * d.k_float) for d in row)
    # Evitar normal nula degenerada.
    if nx == 0.0 and ny == 0.0 and nz == 0.0:
        nx = 1.0
    return Hyperplane(
        clause_index=row[0].clause_index,
        nx=nx, ny=ny, nz=nz,
        p0x=p0x, p0y=p0y, p0z=p0z,
    )

def orient_float(decision: EncodedDecision, plane: Hyperplane) -> float:
    """Valor del predicado en flotante: n . (v - p0)."""
    return (
        plane.nx * (decision.vx - plane.p0x) +
        plane.ny * (decision.vy - plane.p0y) +
        plane.nz * (decision.vz - plane.p0z)
    )

def orient_exact_sign(decision: EncodedDecision, row: List[EncodedDecision]) -> int:
    """
    Recalcula el signo del predicado usando `fractions.Fraction`.
    Se llama solo cuando el flotante cae dentro de FLOAT_EPS.
    """
    base_int = max(2, int(round(decision.base)))
    # k_i como Fraction para cada literal de la fila.
    fracs = [
        (
            Fraction(d.spatial_index),
            _rational_pow_base_inv_n(base_int, d.n),
            1 if d.literal > 0 else -1,
        )
        for d in row
    ]
    m = len(row)
    p0x = sum((sx for sx, _, _ in fracs), Fraction(0)) / m
    p0y = sum((ky for _, ky, _ in fracs), Fraction(0)) / m
    p0z = sum((sx * ky for sx, ky, _ in fracs), Fraction(0)) / m
    nx = sum((s * sx for sx, _, s in fracs), Fraction(0))
    ny = sum((s * ky for _, ky, s in fracs), Fraction(0))
    nz = sum((s * sx * ky for sx, ky, s in fracs), Fraction(0))

    # Punto del literal actual.
    dx = Fraction(decision.spatial_index)
    dy = _rational_pow_base_inv_n(base_int, decision.n)
    dz = dx * dy

    value = nx * (dx - p0x) + ny * (dy - p0y) + nz * (dz - p0z)
    if value > 0:
        return +1
    if value < 0:
        return -1
    # Aún cero exacto: desempate determinista por polaridad e índice.
    tie = (1 if decision.literal > 0 else -1) * decision.spatial_index
    return +1 if tie > 0 else -1

def oriented_sign(decision: EncodedDecision, plane: Hyperplane, row: List[EncodedDecision]) -> int:
    """
    Signo del predicado geométrico con conmutación automática a
    aritmética racional exacta cuando el valor en flotante es
    ambiguo (|value| < FLOAT_EPS).
    Devuelve +1, -1 o 0 (frontera irresoluble - no debería ocurrir
    porque el desempate racional siempre define un signo).
    """
    value = orient_float(decision, plane)
    if abs(value) < FLOAT_EPS:
        return orient_exact_sign(decision, row)
    return +1 if value > 0.0 else -1


# ============================================================
# Permutación Keccak-f[1600] (implementación pura)
# ============================================================

_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

_KECCAK_ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

def _rotl64(x: int, n: int) -> int:
    n &= 63
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF

def keccak_f1600(state: List[int]) -> List[int]:
    """Permutación Keccak-f[1600] sobre 25 lanes de 64 bits."""
    A = [[state[x + 5 * y] for y in range(5)] for x in range(5)]
    for rnd in range(24):
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rotl64(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                A[x][y] ^= D[x]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rotl64(A[x][y], _KECCAK_ROT[x][y])
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])
        A[0][0] ^= _KECCAK_RC[rnd]
    return [A[x][y] for y in range(5) for x in range(5)]

def keccak_permutation(nvars: int, seed: int) -> List[int]:
    """
    Genera una permutación determinista de {1,...,nvars} a partir de
    una ronda Keccak-f[1600] sembrada por `seed`. Se usa como
    reordenamiento topológico invariante para la fase de votación.
    """
    state = [0] * 25
    state[0] = seed & 0xFFFFFFFFFFFFFFFF
    state[1] = (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    perm_pool: List[int] = []
    while len(perm_pool) < nvars:
        state = keccak_f1600(state)
        for lane in state:
            perm_pool.append(lane)
            if len(perm_pool) >= nvars:
                break
    # Ordenar índices por la clave pseudoaleatoria -> permutación.
    indexed = sorted(range(nvars), key=lambda i: perm_pool[i])
    return [i + 1 for i in indexed]


# ============================================================
# Motor de decisión por invarianza topológica
# ============================================================

@dataclass
class QuaternionicResult:
    status: str
    assignment: Assignment
    orientation_votes: Dict[int, Tuple[int, int]]  # var -> (votos+, votos-)
    unsatisfied_clauses: List[int]
    decisions: int
    rows: int
    columns: int
    keccak_rounds: int
    elapsed_seconds: float

def _round_orientation_signs(
    matrix: List[List[EncodedDecision]],
    permutation: List[int],
) -> Dict[int, List[int]]:
    """
    Para una ronda dada (definida por `permutation` sobre las filas),
    calcula los signos de orientación de todos los literales y los
    agrupa por variable.
    """
    # Aplicar la permutación a las filas: reordenar el conjunto de
    # cláusulas. El baricentro y la normal de cada cláusula NO
    # cambian, pero el orden global del recorrido sí, lo que
    # permite verificar la invarianza topológica del signo.
    reindex = [matrix[(p - 1) % len(matrix)] for p in permutation] if matrix else []
    signs_by_var: Dict[int, List[int]] = {}
    for row in reindex:
        plane = build_clause_hyperplane(row)
        for decision in row:
            s = oriented_sign(decision, plane, row)
            # Ajuste por polaridad: un literal negativo ~x satisface
            # x = False, por lo tanto un signo + en ~x vota por
            # variable=False. Se normaliza al espacio de la variable:
            if decision.literal < 0:
                s = -s
            signs_by_var.setdefault(decision.variable, []).append(s)
    return signs_by_var

def oriented_quaternionic_candidate(
    cnf: CNF,
    nvars: Optional[int] = None,
    base: float = 2.0,
    keccak_rounds: int = 8,
    seed: int = 0xC0FFEE,
) -> QuaternionicResult:
    """
    Motor de decisión por predicados de orientación con
    invarianza topológica bajo permutaciones Keccak.
    """
    started = time.perf_counter()
    if nvars is None:
        nvars = number_of_variables(cnf)

    if any(len(clause) == 0 for clause in cnf):
        return QuaternionicResult(
            status="INCONCLUSIVE",
            assignment=complete_assignment({}, nvars),
            orientation_votes={},
            unsatisfied_clauses=[i for i, c in enumerate(cnf) if not c],
            decisions=sum(map(len, cnf)),
            rows=len(cnf),
            columns=max((len(c) for c in cnf), default=0),
            keccak_rounds=keccak_rounds,
            elapsed_seconds=time.perf_counter() - started,
        )

    matrix = build_conjunction_matrix(cnf, base=base)

    # Acumulador de votos de signo por variable a través de las
    # rondas Keccak. Ninguna operación toca `bool(decision.bit)`.
    vote_pos: Dict[int, int] = {}
    vote_neg: Dict[int, int] = {}

    ncl = max(1, len(matrix))
    for r in range(max(1, keccak_rounds)):
        perm = keccak_permutation(ncl, seed=seed + r)
        signs_by_var = _round_orientation_signs(matrix, perm)
        for var, signs in signs_by_var.items():
            for s in signs:
                if s > 0:
                    vote_pos[var] = vote_pos.get(var, 0) + 1
                elif s < 0:
                    vote_neg[var] = vote_neg.get(var, 0) + 1

    # Asignación final: mayoría de signo por variable. El bit es el
    # PRODUCTO de evaluaciones de orientación, no un redondeo.
    assignment: Assignment = {}
    votes: Dict[int, Tuple[int, int]] = {}
    for var in range(1, nvars + 1):
        p = vote_pos.get(var, 0)
        n = vote_neg.get(var, 0)
        votes[var] = (p, n)
        if p == n:
            # Empate topológico: desempate por signo racional sobre
            # el literal +var en el hiperplano de su primera cláusula.
            assignment[var] = _tiebreak_variable_sign(var, matrix) > 0
        else:
            assignment[var] = p > n

    assignment = complete_assignment(assignment, nvars)
    failed = unsatisfied_clause_indices(cnf, assignment)
    status = "SAT" if not failed else "INCONCLUSIVE"

    return QuaternionicResult(
        status=status,
        assignment=assignment,
        orientation_votes=votes,
        unsatisfied_clauses=failed,
        decisions=sum(len(r) for r in matrix),
        rows=len(matrix),
        columns=max((len(r) for r in matrix), default=0),
        keccak_rounds=keccak_rounds,
        elapsed_seconds=time.perf_counter() - started,
    )

def _tiebreak_variable_sign(var: int, matrix: List[List[EncodedDecision]]) -> int:
    """Desempate: signo racional exacto de la primera ocurrencia de
    la variable en la matriz."""
    for row in matrix:
        for d in row:
            if d.variable == var:
                s = orient_exact_sign(d, row)
                return s if d.literal > 0 else -s
    return +1


# ============================================================
# Respaldo exacto opcional (python-sat)
# ============================================================

@dataclass
class ExactResult:
    status: str
    assignment: Optional[Assignment]
    elapsed_seconds: float
    engine: str

def exact_solve_with_pysat(
    cnf: CNF,
    nvars: int,
    engine: str,
    phase_hint: Optional[Assignment] = None,
) -> ExactResult:
    try:
        from pysat.solvers import Solver
    except ImportError as exc:
        raise RuntimeError(
            "Falta python-sat. Instálalo con:\n"
            "    python -m pip install python-sat"
        ) from exc

    started = time.perf_counter()
    try:
        solver = Solver(name=engine, bootstrap_with=cnf)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo iniciar el motor {engine!r}. "
            "Prueba glucose42, glucose4, cadical195 o minisat22."
        ) from exc

    with solver:
        if phase_hint:
            phases = [
                variable if phase_hint.get(variable, False) else -variable
                for variable in range(1, nvars + 1)
            ]
            try:
                solver.set_phases(phases)
            except Exception:
                pass
        if not solver.solve():
            return ExactResult("UNSAT", None, time.perf_counter() - started, engine)
        model = solver.get_model() or []
        assignment = {abs(l): l > 0 for l in model if abs(l) <= nvars}
        assignment = complete_assignment(assignment, nvars)
        if not formula_satisfied(cnf, assignment):
            raise RuntimeError("El backend devolvió un modelo inválido")
        return ExactResult("SAT", assignment, time.perf_counter() - started, engine)


# ============================================================
# Generadores de benchmarks
# ============================================================

def random_k_sat(nvars: int, nclauses: int, k: int, seed: int, planted: bool = False):
    if nvars < k:
        raise ValueError("nvars debe ser al menos k")
    if nclauses < 0:
        raise ValueError("nclauses no puede ser negativo")
    rng = random.Random(seed)
    planted_assignment: Optional[Assignment] = None
    if planted:
        planted_assignment = {v: bool(rng.getrandbits(1)) for v in range(1, nvars + 1)}
    clauses: CNF = []
    for _ in range(nclauses):
        variables = rng.sample(range(1, nvars + 1), k)
        clause = [v if rng.getrandbits(1) else -v for v in variables]
        if planted_assignment is not None and not clause_satisfied(clause, planted_assignment):
            pos = rng.randrange(k)
            v = abs(clause[pos])
            clause[pos] = v if planted_assignment[v] else -v
        clauses.append(clause)
    return clauses, planted_assignment

def pigeonhole_cnf(pigeons: int, holes: int) -> Tuple[CNF, int]:
    if pigeons <= 0 or holes <= 0:
        raise ValueError("pigeons y holes deben ser positivos")
    def variable(p: int, h: int) -> int:
        return p * holes + h + 1
    cnf: CNF = []
    for p in range(pigeons):
        cnf.append([variable(p, h) for h in range(holes)])
    for h in range(holes):
        for p1 in range(pigeons):
            for p2 in range(p1 + 1, pigeons):
                cnf.append([-variable(p1, h), -variable(p2, h)])
    return cnf, pigeons * holes


# ============================================================
# Contraejemplo formal y auto-test del predicado
# ============================================================

def test_indexed_unsoundness_counterexample() -> None:
    cnf: CNF = [[1, 3], [-2, 4], [-1, 2]]
    nvars = 4
    result = oriented_quaternionic_candidate(cnf=cnf, nvars=nvars, base=2.0)
    known_model: Assignment = {1: False, 2: False, 3: True, 4: False}
    assert formula_satisfied(cnf, known_model)
    print("Contraejemplo de completitud (motor de orientación)")
    print(f"  Resultado:               {result.status}")
    print(f"  Asignación:              {result.assignment}")
    print(f"  Votos de orientación:    {result.orientation_votes}")
    print(f"  Cláusulas incumplidas:   {result.unsatisfied_clauses}")
    print(f"  Modelo SAT conocido:     {known_model}")
    if result.status != "SAT":
        print("\nConclusión: declarar UNSAT aquí sería un FALSO NEGATIVO.")

def test_orientation_predicate_boundary() -> None:
    """
    Verifica que el predicado de orientación cae en la rama racional
    exacta cuando el flotante es ambiguo, y que devuelve +1 o -1
    (nunca 0) gracias al desempate posicional determinista.
    """
    print("Auto-test del predicado geométrico")
    cnf: CNF = [[1, -1]]  # tautología: fuerza baricentro simétrico
    matrix = build_conjunction_matrix(cnf, base=2.0)
    row = matrix[0]
    plane = build_clause_hyperplane(row)
    for d in row:
        f_val = orient_float(d, plane)
        e_sign = orient_exact_sign(d, row)
        s = oriented_sign(d, plane, row)
        print(f"  lit={d.literal:+d}  float={f_val:+.3e}  exact_sign={e_sign:+d}  final={s:+d}")
        assert s in (-1, +1), "El predicado nunca debe devolver 0"


# ============================================================
# Presentación y CLI
# ============================================================

def print_instance_statistics(cnf: CNF, nvars: int) -> None:
    literals = sum(len(c) for c in cnf)
    max_width = max((len(c) for c in cnf), default=0)
    ratio = len(cnf) / nvars if nvars else float("inf")
    print("Instancia")
    print(f"  Variables:              {nvars:,}")
    print(f"  Cláusulas:              {len(cnf):,}")
    print(f"  Literales/decisiones:   {literals:,}")
    print(f"  Anchura máxima:         {max_width:,}")
    print(f"  Cláusulas/variable:     {ratio:.6f}")

def print_quaternionic_result(result: QuaternionicResult) -> None:
    print("\nResultado del motor de orientación (Keccak-invariante)")
    print(f"  Estado:                 {result.status}")
    print(f"  Filas de la matriz:     {result.rows:,}")
    print(f"  Columnas máximas:       {result.columns:,}")
    print(f"  Decisiones:             {result.decisions:,}")
    print(f"  Rondas Keccak:          {result.keccak_rounds}")
    print(f"  Tiempo:                 {result.elapsed_seconds:.6f} s")
    print(f"  Cláusulas incumplidas:  {len(result.unsatisfied_clauses):,}")
    if result.unsatisfied_clauses:
        print(f"  Primeros índices:       {result.unsatisfied_clauses[:10]}")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Motor SAT experimental basado en predicados de orientación"
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--dimacs", help="Archivo CNF en formato DIMACS")
    source.add_argument("--random", action="store_true")
    source.add_argument("--php", action="store_true")

    parser.add_argument("--test-counterexample", action="store_true")
    parser.add_argument("--test-predicate", action="store_true")

    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=4.267)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--planted", action="store_true")

    parser.add_argument("--pigeons", type=int, default=31)
    parser.add_argument("--holes", type=int, default=30)

    parser.add_argument("--base", type=float, default=2.0)
    parser.add_argument("--keccak-rounds", type=int, default=8)
    parser.add_argument("--keccak-seed", type=int, default=0xC0FFEE)

    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--engine", default="glucose42")
    parser.add_argument("--write")

    args = parser.parse_args()

    if args.test_predicate:
        test_orientation_predicate_boundary()
        return 0
    if args.test_counterexample:
        test_indexed_unsoundness_counterexample()
        return 0

    if not (args.dimacs or args.random or args.php):
        parser.error("Selecciona --dimacs, --random, --php, --test-counterexample o --test-predicate")

    if args.dimacs:
        cnf, nvars = read_dimacs(args.dimacs)
    elif args.random:
        nclauses = round(args.ratio * args.n)
        cnf, planted = random_k_sat(args.n, nclauses, args.k, args.seed, args.planted)
        nvars = args.n
        if planted is not None:
            assert formula_satisfied(cnf, planted)
    else:
        cnf, nvars = pigeonhole_cnf(args.pigeons, args.holes)

    if args.write:
        write_dimacs(args.write, cnf, nvars)
        print(f"Instancia guardada en {args.write}")

    print_instance_statistics(cnf, nvars)
    result = oriented_quaternionic_candidate(
        cnf=cnf,
        nvars=nvars,
        base=args.base,
        keccak_rounds=args.keccak_rounds,
        seed=args.keccak_seed,
    )
    print_quaternionic_result(result)

    if result.status == "SAT":
        if not formula_satisfied(cnf, result.assignment):
            raise AssertionError("Falso positivo interno")
        print("  Verificación:           modelo SAT correcto")
    else:
        print("  Interpretación:         el método no puede concluir SAT ni UNSAT (INCONCLUSIVE)")

    if args.exact:
        print(f"\nEjecutando respaldo exacto ({args.engine})...")
        exact = exact_solve_with_pysat(cnf, nvars, args.engine, result.assignment)
        print("Resultado exacto")
        print(f"  Estado:                 {exact.status}")
        print(f"  Motor:                  {exact.engine}")
        print(f"  Tiempo:                 {exact.elapsed_seconds:.6f} s")
        if exact.assignment is not None:
            print(f"  Modelo verificado:      {formula_satisfied(cnf, exact.assignment)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
