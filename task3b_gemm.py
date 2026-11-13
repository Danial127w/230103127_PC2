import multiprocessing as mp
import time
from harness import run_benchmark

N = 384
TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def make_matrices(n):
    # Deterministic inputs; Python lists keep the project dependency-free.
    a = [((i * 7) % 101) / 101.0 for i in range(n * n)]
    b = [((i * 11) % 103) / 103.0 for i in range(n * n)]
    return a, b


def gemm_naive(a, b, n):
    c = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i*n+k] * b[k*n+j]
            c[i*n+j] = total
    return c


def gemm_reordered(a, b, n):
    c = [0.0] * (n * n)
    for i in range(n):
        base_i = i * n
        for k in range(n):
            aik = a[base_i + k]
            base_b = k * n
            for j in range(n):
                c[base_i + j] += aik * b[base_b + j]
    return c


def gemm_blocked(a, b, n, bs):
    c = [0.0] * (n * n)
    for ii in range(0, n, bs):
        for kk in range(0, n, bs):
            for jj in range(0, n, bs):
                i_end = min(ii + bs, n)
                k_end = min(kk + bs, n)
                j_end = min(jj + bs, n)
                for i in range(ii, i_end):
                    base_i = i * n
                    for k in range(kk, k_end):
                        aik = a[base_i + k]
                        base_b = k * n
                        for j in range(jj, j_end):
                            c[base_i + j] += aik * b[base_b + j]
    return c


def max_difference(x, y):
    return max(abs(a - b) for a, b in zip(x, y))


def report(label, fn):
    stats = run_benchmark(fn, warmup_runs=0, timed_runs=1)
    gflops = (2 * N**3) / (stats.median_ms * 1e6)
    print(f"{label}: {stats.median_ms:.3f} ms, {gflops:.5f} GFLOP/s")
    return stats.median_ms, gflops


def main():
    print("PHASE 3B: GEMM")
    a, b = make_matrices(N)
    t1, g1 = report("Variant 1 naive", lambda: gemm_naive(a, b, N))
    c2 = gemm_reordered(a, b, N)
    t2 = None
    print("Variant 2 calculated; comparing variants...")
    c1 = gemm_naive(a, b, N)
    print("max |C1-C2|:", max_difference(c1, c2))
    print(f"Variant 1: {t1:.3f} ms, {g1:.5f} GFLOP/s")
    print("\nBlocked tile sweep:")
    best = None
    for bs in TILE_SIZES:
        elapsed, gflops = report(f"BS={bs}", lambda bs=bs: gemm_blocked(a, b, N, bs))
        if best is None or gflops > best[1]:
            best = (bs, gflops)
    print(f"Best tile size: {best[0]}, throughput={best[1]:.5f} GFLOP/s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
