import ctypes
import multiprocessing as mp
import time
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103127
BYTES = 64 * 1024 * 1024
N = 256


def zero_memory(buffer, size):
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    ctypes.memset(address, 0, size)


def fill_memory(buffer, value, size):
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    ctypes.memset(address, value, size)


def cp(a, b, lo, hi):
    x = shared_memory.SharedMemory(name=a)
    y = shared_memory.SharedMemory(name=b)

    x.buf[lo:hi] = y.buf[lo:hi]

    x.close()
    y.close()


def compute(seed, iters):
    x = [0.1 + seed * 0.001 + j * 0.01 for j in range(16)]

    for i in range(iters):
        for j in range(16):
            x[j] = x[j] * 1.0000001 + x[(j + 1) & 15] * 0.0000001

    return sum(x)


def roof():
    p = mp.cpu_count()

    a = shared_memory.SharedMemory(create=True, size=BYTES)
    b = shared_memory.SharedMemory(create=True, size=BYTES)

    try:
        zero_memory(a.buf, BYTES)
        fill_memory(b.buf, 1, BYTES)

        c = (BYTES + p - 1) // p

        jobs = [
            (a.name, b.name, i * c, min(BYTES, (i + 1) * c))
            for i in range(p)
        ]

        def copy():
            ps = [
                mp.Process(target=cp, args=j)
                for j in jobs
            ]

            for q in ps:
                q.start()

            for q in ps:
                q.join()

            if any(q.exitcode for q in ps):
                raise RuntimeError("Copy worker failed")

        cs = run_benchmark(copy, 1, 3)
        bw = 2 * BYTES / (cs.median_ms * 1e6)

        it = 1_875_000

        def calc():
            ps = [
                mp.Process(target=compute, args=(w, it))
                for w in range(p)
            ]

            for q in ps:
                q.start()

            for q in ps:
                q.join()

            if any(q.exitcode for q in ps):
                raise RuntimeError("Compute worker failed")

        ts = run_benchmark(calc, 0, 3)

        gf = p * it * 16 * 2 / (ts.median_ms * 1e6)

        print(
            f"Bandwidth={bw:.3f} GB/s; "
            f"compute={gf:.3f} GFLOP/s; "
            f"ridge={gf / bw:.3f} FLOP/byte"
        )

        return bw, gf

    finally:
        a.close()
        a.unlink()
        b.close()
        b.unlink()


def gemm_worker(an, bn, cn, lo, hi, v, bs):
    sa = shared_memory.SharedMemory(name=an)
    sb = shared_memory.SharedMemory(name=bn)
    sc = shared_memory.SharedMemory(name=cn)

    A = (ctypes.c_double * (N * N)).from_buffer(sa.buf)
    B = (ctypes.c_double * (N * N)).from_buffer(sb.buf)
    C = (ctypes.c_double * (N * N)).from_buffer(sc.buf)

    if v == 1:
        for i in range(lo, hi):
            for j in range(N):
                z = 0.0

                for k in range(N):
                    z += A[i * N + k] * B[k * N + j]

                C[i * N + j] = z

    elif v == 2:
        for i in range(lo, hi):
            base = i * N

            for k in range(N):
                x = A[base + k]
                bk = k * N

                for j in range(N):
                    C[base + j] += x * B[bk + j]

    else:
        for ii in range(lo, hi, bs):
            for kk in range(0, N, bs):
                for jj in range(0, N, bs):
                    for i in range(ii, min(hi, ii + bs)):
                        base = i * N

                        for k in range(kk, min(N, kk + bs)):
                            x = A[base + k]
                            bk = k * N

                            for j in range(jj, min(N, jj + bs)):
                                C[base + j] += x * B[bk + j]

    del A, B, C

    sa.close()
    sb.close()
    sc.close()


def run_gemm(v, bs=32):
    p = mp.cpu_count()

    a = shared_memory.SharedMemory(create=True, size=N * N * 8)
    b = shared_memory.SharedMemory(create=True, size=N * N * 8)
    c = shared_memory.SharedMemory(create=True, size=N * N * 8)

    try:
        A = (ctypes.c_double * (N * N)).from_buffer(a.buf)
        B = (ctypes.c_double * (N * N)).from_buffer(b.buf)

        for i in range(N):
            for j in range(N):
                A[i * N + j] = ((i + j) % 17) * 0.01
                B[i * N + j] = ((i * 3 + j) % 19) * 0.01

        del A, B

        row = (N + p - 1) // p

        zero_memory(c.buf, N * N * 8)

        ps = []
        t = time.perf_counter_ns()

        for w in range(p):
            lo = w * row
            hi = min(N, (w + 1) * row)

            if lo < hi:
                q = mp.Process(
                    target=gemm_worker,
                    args=(a.name, b.name, c.name, lo, hi, v, bs)
                )

                q.start()
                ps.append(q)

        for q in ps:
            q.join()

            if q.exitcode:
                raise RuntimeError("GEMM worker failed")

        ms = (time.perf_counter_ns() - t) / 1e6

        out = (ctypes.c_double * (N * N)).from_buffer(c.buf)
        result = list(out)

        del out

        return ms, result

    finally:
        a.close()
        a.unlink()

        b.close()
        b.unlink()

        c.close()
        c.unlink()


def main():
    print("PHASE 3A: ROOFLINE")

    roof()

    print("PHASE 3B: GEMM N=256")

    t1, o1 = run_gemm(1)
    t2, o2 = run_gemm(2)

    print(
        f"Variant1 {t1:.3f} ms "
        f"{2 * N**3 / (t1 * 1e6):.4f} GFLOP/s"
    )

    print(
        f"Variant2 {t2:.3f} ms "
        f"{2 * N**3 / (t2 * 1e6):.4f} GFLOP/s"
    )

    best = None
    g1 = 2 * N**3 / (t1 * 1e6)

    for bs in (8, 16, 32, 64, 96, 128):
        t, o = run_gemm(3, bs)

        err = max(
            abs(x - y)
            for x, y in zip(o2, o)
        )

        g = 2 * N**3 / (t * 1e6)

        print(
            f"BS={bs} "
            f"time={t:.3f} ms "
            f"GFLOP/s={g:.4f} "
            f"footprint={3 * bs * bs * 8} B "
            f"intensity~{bs / 12:.3f} "
            f"error={err:.3e}"
        )

        if best is None or g > best[0]:
            best = (g, bs)

    check_time, check_result = run_gemm(3, best[1])

    if max(
        abs(x - y)
        for x, y in zip(o2, check_result)
    ) >= 1e-9:
        raise AssertionError("GEMM check failed")

    print("Best tile:", best[1])

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            g1,
            best[0]
        )
    )


if __name__ == "__main__":
    main()