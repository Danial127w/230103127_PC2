import multiprocessing as mp
import math
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103127
N_STRONG = 1_600_000
N0_WEAK = 250_000
PROCESS_COUNTS = list(range(1, mp.cpu_count() + 1))


def serial_stage(n):
    # Serial prefix: first 5% of N, 12 sin/sqrt operations per element.
    total = 0.0
    for i in range(max(1, int(0.05 * n))):
        x = (i + 1) * 0.0001
        for j in range(12):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        total += x
    return total


def _parallel_range(bounds):
    start, end = bounds
    total = 0.0
    for i in range(start, end):
        x = (i + 1) * 0.00001
        for j in range(35):
            x = x * 1.0000001 + 0.0000003
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        total += x
    return total


def parallel_worker(args):
    return _parallel_range(args)


def make_ranges(n, p):
    # Split Stage B's full N range into p balanced contiguous chunks.
    return [(i * n // p, (i + 1) * n // p) for i in range(p)]


def run_pipeline(pool, n, p):
    serial_value = serial_stage(n)
    partials = pool.map(parallel_worker, make_ranges(n, p))
    return serial_value + sum(partials)


def timed_pipeline(n, p, repeats=5):
    # Pool creation is outside timed region, as required by the handout.
    with mp.Pool(processes=p) as pool:
        return run_benchmark(lambda: run_pipeline(pool, n, p),
                             warmup_runs=1, timed_runs=repeats)


def main():
    print("PHASE 2: STRONG AND WEAK SCALING")
    strong = {}
    print("\nSTRONG SCALING: p,N,median_ms,speedup,efficiency")
    for p in PROCESS_COUNTS:
        stats = timed_pipeline(N_STRONG, p)
        strong[p] = stats.median_ms
    t1 = strong[1]
    for p in PROCESS_COUNTS:
        speedup = t1 / strong[p]
        print(f"{p},{N_STRONG},{strong[p]:.3f},{speedup:.4f},"
              f"{speedup / p:.4f}")

    # Measure serial stage independently at p=1 for alpha estimate.
    serial_stats = run_benchmark(lambda: serial_stage(N_STRONG),
                                 warmup_runs=1, timed_runs=5)
    serial_fraction = serial_stats.median_ms / strong[1]
    amdahl_limit = 1 / serial_fraction if serial_fraction > 0 else float("inf")
    pmax = PROCESS_COUNTS[-1]
    amdahl_at_max = 1 / (serial_fraction + (1 - serial_fraction) / pmax)
    print(f"\nSerial fraction s={serial_fraction:.6f}")
    print(f"Amdahl theoretical maximum={amdahl_limit:.4f}")
    print(f"Amdahl speedup at p={pmax}: {amdahl_at_max:.4f}")
    print(f"Observed strong speedup at p={pmax}: {t1 / strong[pmax]:.4f}")

    weak = {}
    weak_alpha = {}
    print("\nWEAK SCALING: p,N,median_ms,efficiency,Gustafson_speedup")
    for p in PROCESS_COUNTS:
        n = p * N0_WEAK
        with mp.Pool(processes=p) as pool:
            stats = run_benchmark(lambda: run_pipeline(pool, n, p),
                                  warmup_runs=1, timed_runs=5)
        weak[p] = stats.median_ms
        stage_a = run_benchmark(lambda: serial_stage(n),
                                warmup_runs=1, timed_runs=3).median_ms
        alpha = stage_a / stats.median_ms
        weak_alpha[p] = alpha
        gustafson = p - alpha * (p - 1)
        print(f"{p},{n},{stats.median_ms:.3f},"
              f"{weak[1] / stats.median_ms:.4f},{gustafson:.4f}")

    print(emit_verify_tag("PHASE2", STUDENT_ID,
                          t1 / strong[pmax],
                          pmax - weak_alpha[pmax] * (pmax - 1)))


if __name__ == "__main__":
    mp.freeze_support()
    main()
