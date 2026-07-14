# -*- coding: utf-8 -*-
"""quaternionic_sat_fixed.py

Versión CORREGIDA del prototipo experimental del "algoritmo P=NP"
descrito en el documento.

Correcciones aplicadas (respecto de la versión original):

1. La codificación de cada literal incluye ahora un ÍNDICE ESPACIAL x
   global, y la conjunción indexada es

       K = sum_{x=1}^{L} x * k_x,     k_x = base^(1/n_x)

   con derivadas

       dk_x/dn_x = -base^(1/n_x) * ln(base) / n_x^2
       dK/dn_x   =  x * dk_x/dn_x.

2. El ordenamiento de filas usa el gradiente ESPACIAL INDEXADO
   (dK/dn_x), no solo la derivada local.

3. Se elimina cualquier salto lógico del tipo
       "asignación fallida  =>  UNSAT".
   Una proyección rechazada NO certifica insatisfacibilidad:
   el estado pasa a ser "INCONCLUSIVE".

4. Se añade una prueba automática (test_indexed_unsoundness_counterexample)
   que exhibe una fórmula SAT cuya proyección longitudinal cae en una
   asignación que no la satisface. Sirve como contraejemplo formal a
   la regla ingenua "UNKNOWN -> UNSAT".

5. La arquitectura segura sigue siendo:
       - "SAT" solo si la asignación se VERIFICA sobre la fórmula.
       - "UNSAT" solo si un backend exacto (python-sat) lo certifica.
       - "INCONCLUSIVE" en cualquier otro caso.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Clause = List[int]
CNF = List[Clause]
Assignment = Dict[int, bool]


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


def unsatisfied_clause_indices(
    cnf: CNF,
    assignment: Assignment,
) -> List[int]:
    return [
        i for i, clause in enumerate(cnf)
        if not clause_satisfied(clause, assignment)
    ]


def complete_assignment(
    assignment: Assignment,
    nvars: int,
    default: bool = False,
) -> Assignment:
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

    actual_nvars = number_of_variables(clauses)
    return clauses, max(declared_nvars, actual_nvars)


def write_dimacs(path: str, cnf: CNF, nvars: Optional[int] = None) -> None:
    if nvars is None:
        nvars = number_of_variables(cnf)

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"p cnf {nvars} {len(cnf)}\n")
        for clause in cnf:
            file.write(" ".join(map(str, clause)) + " 0\n")


# ============================================================
# Algoritmo del documento — versión INDEXADA
# ============================================================

@dataclass(frozen=True)
class EncodedDecision:
    clause_index: int
    occurrence_index: int
    spatial_index: int
    literal: int
    variable: int
    bit: int
    n: int
    a: float
    b: float
    base: float
    k: float
    unindexed_derivative: float
    indexed_derivative: float


@dataclass
class QuaternionicResult:
    status: str
    assignment: Assignment
    binary_sequence: List[int]
    unsatisfied_clauses: List[int]
    decisions: int
    rows: int
    columns: int
    elapsed_seconds: float


def encode_literal(
    literal: int,
    clause_index: int,
    occurrence_index: int,
    spatial_index: int,
    base: float = 2.0,
) -> EncodedDecision:
    """
    Conjunción indexada:

        K       = sum_x  x * k_x
        k_x     = base ** (1 / n_x)
        n_x     = literal firmado

    Derivadas:

        dk_x/dn_x = -base**(1/n_x) * ln(base) / n_x**2
        dK/dn_x   =  x * dk_x/dn_x
    """
    if literal == 0:
        raise ValueError("Un literal SAT no puede ser cero")
    if spatial_index < 1:
        raise ValueError("El índice espacial debe comenzar en 1")
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError(
            "La base a+b debe ser positiva, finita y distinta de 1"
        )

    n = literal
    a = 1.0
    b = base - 1.0
    k = math.exp(math.log(base) / n)
    unindexed_derivative = -(k * math.log(base)) / (n * n)
    indexed_derivative = spatial_index * unindexed_derivative

    return EncodedDecision(
        clause_index=clause_index,
        occurrence_index=occurrence_index,
        spatial_index=spatial_index,
        literal=literal,
        variable=abs(literal),
        bit=1 if literal > 0 else 0,
        n=n,
        a=a,
        b=b,
        base=base,
        k=k,
        unindexed_derivative=unindexed_derivative,
        indexed_derivative=indexed_derivative,
    )


def build_conjunction_matrix(
    cnf: CNF,
    base: float = 2.0,
) -> List[List[EncodedDecision]]:
    """
    Construye la matriz de la serie indexada.

    Los índices espaciales se asignan globalmente en orden:
        cláusula 0, literal 0  -> x = 1
        cláusula 0, literal 1  -> x = 2
        ...
        cláusula 1, literal 0  -> x = k+1
        ...

    La conjunción representada es:

        K = sum_{x=1}^{L} x * k_x
    """
    matrix: List[List[EncodedDecision]] = []
    spatial_index = 1
    for clause_index, clause in enumerate(cnf):
        row: List[EncodedDecision] = []
        for occurrence_index, literal in enumerate(clause):
            decision = encode_literal(
                literal=literal,
                clause_index=clause_index,
                occurrence_index=occurrence_index,
                spatial_index=spatial_index,
                base=base,
            )
            row.append(decision)
            spatial_index += 1
        matrix.append(row)
    return matrix


def sort_matrix_rows(
    matrix: List[List[EncodedDecision]],
) -> List[List[EncodedDecision]]:
    """
    Orden ascendente de izquierda a derecha usando el gradiente
    ESPACIAL INDEXADO.  Se rompen empates de forma determinista.
    """
    return [
        sorted(
            row,
            key=lambda decision: (
                decision.indexed_derivative,
                decision.spatial_index,
                decision.variable,
                decision.literal,
            ),
        )
        for row in matrix
    ]


def longitudinal_read(
    matrix: List[List[EncodedDecision]],
) -> List[EncodedDecision]:
    """
    Lectura longitudinal:
        primero columna 0 de todas las filas,
        después columna 1 de todas las filas,
        etc.
    """
    max_columns = max((len(row) for row in matrix), default=0)
    sequence: List[EncodedDecision] = []
    for column in range(max_columns):
        for row in matrix:
            if column < len(row):
                sequence.append(row[column])
    return sequence


def binary_sequence_to_assignment(
    decisions: Sequence[EncodedDecision],
    nvars: int,
) -> Tuple[List[int], Assignment]:
    """
    Traduce la lectura longitudinal a una secuencia binaria
    y a una asignación completa.

    Regla: la PRIMERA aparición longitudinal de una variable
    fija su bit.  Las variables nunca observadas reciben False.
    """
    binary_sequence = [decision.bit for decision in decisions]
    assignment: Assignment = {}
    for decision in decisions:
        if decision.variable not in assignment:
            assignment[decision.variable] = bool(decision.bit)
    return binary_sequence, complete_assignment(
        assignment,
        nvars=nvars,
        default=False,
    )


def indexed_quaternionic_candidate(
    cnf: CNF,
    nvars: Optional[int] = None,
    base: float = 2.0,
) -> QuaternionicResult:
    """
    Ejecuta fielmente la versión INDEXADA del procedimiento matricial.

    * "SAT"          <=> la asignación se verifica sobre la fórmula.
    * "INCONCLUSIVE" <=> la asignación no la satisface.  Esto NO
                         demuestra UNSAT: existen fórmulas SAT que
                         caen en este caso (ver contraejemplo).
    """
    started = time.perf_counter()
    if nvars is None:
        nvars = number_of_variables(cnf)

    # Cláusula vacía => insatisfacible por definición sintáctica,
    # pero no por el método matricial; se reporta como INCONCLUSIVE.
    if any(len(clause) == 0 for clause in cnf):
        return QuaternionicResult(
            status="INCONCLUSIVE",
            assignment=complete_assignment({}, nvars),
            binary_sequence=[],
            unsatisfied_clauses=[
                i for i, clause in enumerate(cnf) if not clause
            ],
            decisions=sum(map(len, cnf)),
            rows=len(cnf),
            columns=max((len(c) for c in cnf), default=0),
            elapsed_seconds=time.perf_counter() - started,
        )

    matrix = build_conjunction_matrix(cnf, base=base)
    ordered_matrix = sort_matrix_rows(matrix)
    longitudinal_sequence = longitudinal_read(ordered_matrix)
    binary_sequence, assignment = binary_sequence_to_assignment(
        longitudinal_sequence,
        nvars=nvars,
    )

    failed = unsatisfied_clause_indices(cnf, assignment)

    # SAT sí está certificado por evaluación directa.
    # Una asignación rechazada NO certifica UNSAT.
    status = "SAT" if not failed else "INCONCLUSIVE"

    return QuaternionicResult(
        status=status,
        assignment=assignment,
        binary_sequence=binary_sequence,
        unsatisfied_clauses=failed,
        decisions=sum(len(row) for row in matrix),
        rows=len(matrix),
        columns=max((len(row) for row in matrix), default=0),
        elapsed_seconds=time.perf_counter() - started,
    )


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
    """
    Respaldo exacto mediante python-sat.  Es la única manera legítima
    de certificar UNSAT en este prototipo.
    """
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

        satisfiable = solver.solve()

        if not satisfiable:
            return ExactResult(
                status="UNSAT",
                assignment=None,
                elapsed_seconds=time.perf_counter() - started,
                engine=engine,
            )

        model = solver.get_model() or []
        assignment = {
            abs(literal): literal > 0
            for literal in model
            if abs(literal) <= nvars
        }
        assignment = complete_assignment(assignment, nvars)

        if not formula_satisfied(cnf, assignment):
            raise RuntimeError(
                "El backend devolvió un modelo que no supera la verificación"
            )

        return ExactResult(
            status="SAT",
            assignment=assignment,
            elapsed_seconds=time.perf_counter() - started,
            engine=engine,
        )


# ============================================================
# Generadores de benchmarks
# ============================================================

def random_k_sat(
    nvars: int,
    nclauses: int,
    k: int,
    seed: int,
    planted: bool = False,
) -> Tuple[CNF, Optional[Assignment]]:
    if nvars < k:
        raise ValueError("nvars debe ser al menos k")
    if nclauses < 0:
        raise ValueError("nclauses no puede ser negativo")

    rng = random.Random(seed)

    planted_assignment: Optional[Assignment] = None
    if planted:
        planted_assignment = {
            variable: bool(rng.getrandbits(1))
            for variable in range(1, nvars + 1)
        }

    clauses: CNF = []
    for _ in range(nclauses):
        variables = rng.sample(range(1, nvars + 1), k)
        clause = [
            variable if rng.getrandbits(1) else -variable
            for variable in variables
        ]
        if planted_assignment is not None:
            if not clause_satisfied(clause, planted_assignment):
                position = rng.randrange(k)
                variable = abs(clause[position])
                clause[position] = (
                    variable if planted_assignment[variable] else -variable
                )
        clauses.append(clause)

    return clauses, planted_assignment


def pigeonhole_cnf(pigeons: int, holes: int) -> Tuple[CNF, int]:
    if pigeons <= 0 or holes <= 0:
        raise ValueError("pigeons y holes deben ser positivos")

    def variable(pigeon: int, hole: int) -> int:
        return pigeon * holes + hole + 1

    cnf: CNF = []
    for pigeon in range(pigeons):
        cnf.append([
            variable(pigeon, hole)
            for hole in range(holes)
        ])
    for hole in range(holes):
        for p1 in range(pigeons):
            for p2 in range(p1 + 1, pigeons):
                cnf.append([
                    -variable(p1, hole),
                    -variable(p2, hole),
                ])
    return cnf, pigeons * holes


# ============================================================
# Contraejemplo formal
# ============================================================

def test_indexed_unsoundness_counterexample() -> None:
    """
    Demuestra que una proyección longitudinal fallida NO implica UNSAT.

    Fórmula:
        (x1 v x3) ^ (~x2 v x4) ^ (~x1 v x2)

    Es SAT (por ejemplo x1=0, x2=0, x3=1, x4=0),
    pero la proyección indexada puede caer en una asignación que
    incumple la tercera cláusula.
    """
    cnf: CNF = [
        [1, 3],
        [-2, 4],
        [-1, 2],
    ]
    nvars = 4

    result = indexed_quaternionic_candidate(
        cnf=cnf,
        nvars=nvars,
        base=2.0,
    )

    known_model: Assignment = {
        1: False,
        2: False,
        3: True,
        4: False,
    }
    assert formula_satisfied(cnf, known_model), (
        "El modelo conocido debería satisfacer la fórmula"
    )

    print("Contraejemplo de completitud")
    print(f"  Resultado matricial:     {result.status}")
    print(f"  Asignación matricial:    {result.assignment}")
    print(f"  Cláusulas incumplidas:   {result.unsatisfied_clauses}")
    print(f"  Modelo SAT conocido:     {known_model}")
    print(
        "  Modelo verificado:       "
        f"{formula_satisfied(cnf, known_model)}"
    )

    if result.status != "SAT":
        print(
            "\nConclusión: declarar UNSAT aquí sería un FALSO NEGATIVO.\n"
            "Una asignación rechazada no certifica insatisfacibilidad."
        )
    else:
        print(
            "\nEn esta ejecución la proyección sí satisfizo la fórmula.\n"
            "El contraejemplo sigue siendo válido en general porque la\n"
            "proyección depende del orden sintáctico de cláusulas/literales."
        )


# ============================================================
# Presentación y ejecución
# ============================================================

def print_instance_statistics(cnf: CNF, nvars: int) -> None:
    literals = sum(len(clause) for clause in cnf)
    max_width = max((len(clause) for clause in cnf), default=0)
    ratio = len(cnf) / nvars if nvars else float("inf")

    print("Instancia")
    print(f"  Variables:              {nvars:,}")
    print(f"  Cláusulas:              {len(cnf):,}")
    print(f"  Literales/decisiones:   {literals:,}")
    print(f"  Anchura máxima:         {max_width:,}")
    print(f"  Cláusulas/variable:     {ratio:.6f}")


def print_quaternionic_result(result: QuaternionicResult) -> None:
    print("\nResultado del algoritmo del documento (versión indexada)")
    print(f"  Estado:                 {result.status}")
    print(f"  Filas de la matriz:     {result.rows:,}")
    print(f"  Columnas máximas:       {result.columns:,}")
    print(f"  Decisiones:             {result.decisions:,}")
    print(f"  Tiempo:                 {result.elapsed_seconds:.6f} s")
    print(
        "  Cláusulas incumplidas:  "
        f"{len(result.unsatisfied_clauses):,}"
    )
    if result.unsatisfied_clauses:
        preview = result.unsatisfied_clauses[:10]
        print(f"  Primeros índices:       {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Implementación experimental (corregida) del algoritmo "
            "matricial SAT"
        )
    )

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--dimacs", help="Archivo CNF en formato DIMACS")
    source.add_argument(
        "--random",
        action="store_true",
        help="Generar k-SAT aleatorio",
    )
    source.add_argument(
        "--php",
        action="store_true",
        help="Generar principio del palomar",
    )

    parser.add_argument(
        "--test-counterexample",
        action="store_true",
        help="Ejecutar el contraejemplo SAT de la proyección única",
    )

    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=4.267)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--planted", action="store_true")

    parser.add_argument("--pigeons", type=int, default=31)
    parser.add_argument("--holes", type=int, default=30)

    parser.add_argument(
        "--base",
        type=float,
        default=2.0,
        help="Valor de a_j+b_j; por defecto 2",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Ejecutar también un SAT solver exacto",
    )
    parser.add_argument(
        "--engine",
        default="glucose42",
        help="Backend python-sat",
    )
    parser.add_argument(
        "--write",
        help="Guardar la instancia generada en DIMACS",
    )

    args = parser.parse_args()

    if args.test_counterexample:
        test_indexed_unsoundness_counterexample()
        return 0

    if not (args.dimacs or args.random or args.php):
        parser.error(
            "Selecciona --dimacs, --random, --php o --test-counterexample"
        )

    if args.dimacs:
        cnf, nvars = read_dimacs(args.dimacs)
    elif args.random:
        nclauses = round(args.ratio * args.n)
        cnf, planted_assignment = random_k_sat(
            nvars=args.n,
            nclauses=nclauses,
            k=args.k,
            seed=args.seed,
            planted=args.planted,
        )
        nvars = args.n
        if planted_assignment is not None:
            assert formula_satisfied(cnf, planted_assignment)
    else:
        cnf, nvars = pigeonhole_cnf(
            pigeons=args.pigeons,
            holes=args.holes,
        )

    if args.write:
        write_dimacs(args.write, cnf, nvars)
        print(f"Instancia guardada en {args.write}")

    print_instance_statistics(cnf, nvars)

    result = indexed_quaternionic_candidate(
        cnf=cnf,
        nvars=nvars,
        base=args.base,
    )
    print_quaternionic_result(result)

    if result.status == "SAT":
        if not formula_satisfied(cnf, result.assignment):
            raise AssertionError("Falso positivo interno")
        print("  Verificación:           modelo SAT correcto")
    else:
        print(
            "  Interpretación:         el método no puede concluir "
            "SAT ni UNSAT (INCONCLUSIVE)"
        )

    if args.exact:
        print(f"\nEjecutando respaldo exacto ({args.engine})...")
        exact = exact_solve_with_pysat(
            cnf=cnf,
            nvars=nvars,
            engine=args.engine,
            phase_hint=result.assignment,
        )
        print("Resultado exacto")
        print(f"  Estado:                 {exact.status}")
        print(f"  Motor:                  {exact.engine}")
        print(f"  Tiempo:                 {exact.elapsed_seconds:.6f} s")
        if exact.assignment is not None:
            verified = formula_satisfied(cnf, exact.assignment)
            print(f"  Modelo verificado:      {verified}")
        if result.status != "SAT":
            print(
                "  Comparación:            la parte exacta fue necesaria"
            )
        else:
            print(
                "  Comparación:            el algoritmo matricial "
                "encontró un modelo por sí mismo"
            )

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
