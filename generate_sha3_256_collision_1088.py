#!/usr/bin/env python3
"""Emit a pure DIMACS CNF encoding for two 1088-bit SHA3-256 inputs."""
from pathlib import Path

OUT = Path("sha3_256_collision_1088.cnf")
BODY = Path(".sha3_256_collision_1088.body")

# Keccak-f[1600] constants; bit 0 is the least-significant bit of a lane.
RC = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
]
RHO = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)

next_var = 0
nclauses = 0
buf = []

with BODY.open("w", encoding="ascii", buffering=8 * 1024 * 1024) as f:
    def flush():
        nonlocal_buf = None  # harmless local marker for clarity in generated source
        if buf:
            f.write("".join(buf))
            buf.clear()

    def newvar():
        global next_var
        next_var += 1
        return next_var

    def clause(*lits):
        global nclauses
        nclauses += 1
        buf.append(" ".join(map(str, lits)) + " 0\n")
        if len(buf) >= 16384:
            flush()

    def neg(lit):
        return -lit

    # The only constant wire: Boolean false.
    zero = newvar()
    clause(-zero)

    # Variables 2..1089 are M1, and 1090..2177 are M2 (bits are LSB-first per byte).
    m1 = [newvar() for _ in range(1088)]
    m2 = [newvar() for _ in range(1088)]

    def xor(a, b):
        """Allocate z with z iff (a xor b), for signed input literals."""
        z = newvar()
        clause(-a, -b, -z)
        clause(a, b, -z)
        clause(-a, b, z)
        clause(a, -b, z)
        return z

    def and_(a, b):
        """Allocate z with z iff (a and b), for signed input literals."""
        z = newvar()
        clause(-a, -b, z)
        clause(a, -z)
        clause(b, -z)
        return z

    def permute(state):
        """Exact 24-round Keccak-f[1600]; state keys are (x,y,z), literals signed."""
        for rnd in range(24):
            c = {}
            for x in range(5):
                for z in range(64):
                    acc = state[(x, 0, z)]
                    for y in range(1, 5):
                        acc = xor(acc, state[(x, y, z)])
                    c[(x, z)] = acc

            d = {}
            for x in range(5):
                for z in range(64):
                    d[(x, z)] = xor(c[((x - 1) % 5, z)],
                                    c[((x + 1) % 5, (z - 1) % 64)])

            theta = {}
            for x in range(5):
                for y in range(5):
                    for z in range(64):
                        theta[(x, y, z)] = xor(state[(x, y, z)], d[(x, z)])

            b = {}
            for x in range(5):
                for y in range(5):
                    for z in range(64):
                        b[(y, (2 * x + 3 * y) % 5, (z + RHO[x][y]) % 64)] = theta[(x, y, z)]

            state = {}
            for x in range(5):
                for y in range(5):
                    for z in range(64):
                        t = and_(neg(b[((x + 1) % 5, y, z)]), b[((x + 2) % 5, y, z)])
                        state[(x, y, z)] = xor(b[(x, y, z)], t)

            rc = RC[rnd]
            for z in range(64):
                if (rc >> z) & 1:
                    state[(0, 0, z)] = neg(state[(0, 0, z)])
        return state

    def state_from_block(message):
        s = {}
        for x in range(5):
            for y in range(5):
                for z in range(64):
                    i = (x + 5 * y) * 64 + z
                    s[(x, y, z)] = message[i] if i < 1088 else zero
        return s

    def absorb_sha3_256_one_rate_block(message):
        # The input message is exactly one full 1088-bit SHA3-256 rate block.
        # SHA-3's suffix/pad block has 1 bits at bit positions 1, 2, and 1087.
        s = permute(state_from_block(message))
        for x in range(5):
            for y in range(5):
                for z in range(64):
                    i = (x + 5 * y) * 64 + z
                    if i in (1, 2, 1087):
                        s[(x, y, z)] = neg(s[(x, y, z)])
        return permute(s)

    h1state = absorb_sha3_256_one_rate_block(m1)
    h2state = absorb_sha3_256_one_rate_block(m2)

    # First 256 squeezed bits must agree, allowing signed literals after iota/padding.
    for i in range(256):
        x, rem = divmod(i, 64)
        z = rem
        a = h1state[(x, 0, z)]
        b = h2state[(x, 0, z)]
        clause(-a, b)
        clause(a, -b)

    # Require a strict difference between the two 1088-bit message variables.
    differences = [xor(a, b) for a, b in zip(m1, m2)]
    clause(*differences)
    flush()

with OUT.open("w", encoding="ascii", buffering=8 * 1024 * 1024) as out, BODY.open("r", encoding="ascii", buffering=8 * 1024 * 1024) as body:
    out.write(f"p cnf {next_var} {nclauses}\n")
    while True:
        chunk = body.read(8 * 1024 * 1024)
        if not chunk:
            break
        out.write(chunk)

BODY.unlink()
print(f"p cnf {next_var} {nclauses}")
