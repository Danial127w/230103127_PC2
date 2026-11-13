import multiprocessing as mp
import math
import time
from harness import emit_verify_tag

STUDENT_ID = 230103127
N = 12_000
CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]


def workload(i):
    k = 1 + (i // 60) ** 2
    x = (i + 1) * 0.00001
    acc = 0.0
    for j in range(k):
        acc += math.sin(x + j * 0.0001) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return acc


def _timed_items(indices):
    start = time.perf_counter()
    total = 0.0
    for i in indices:
        total += workload(i)
    return time.perf_counter() - start, total


def _static_worker(args):
    start, end = args
    return _timed_items(range(start, end))


def _cyclic_worker(args):
    worker, p, n = args
    return _timed_items(range(worker, n, p))


def _dynamic_worker(args):
    # Worker claims chunks under a lock from shared integer counter.
    counter, lock, n, chunk = args
    elapsed_start = time.perf_counter()
    total = 0.0
    while True:
        with lock:
            start = counter.value
            counter.value += chunk
        if start >= n:
            break
        for i in range(start, min(start + chunk, n)):
            total += workload(i)
    return time.perf_counter() - elapsed_start, total


def run_static(p):
    bounds = [(i * N // p, (i + 1) * N // p) for i in range(p)]
    wall0 = time.perf_counter()
    with mp.Pool(p) as pool:
        results = pool.map(_static_worker, bounds)
    return (time.perf_counter() - wall0) * 1000, [r[0] * 1000 for r in results]


def run_cyclic(p):
    args = [(i, p, N) for i in range(p)]
    wall0 = time.perf_counter()
    with mp.Pool(p) as pool:
        results = pool.map(_cyclic_worker, args)
    return (time.perf_counter() - wall0) * 1000, [r[0] * 1000 for r in results]


def run_dynamic(p, chunk):
    # Manager proxies can safely be passed to Pool tasks under macOS spawn.
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        args = [(counter, lock, N, chunk)] * p
        wall0 = time.perf_counter()
        with mp.Pool(p) as pool:
            results = pool.map(_dynamic_worker, args)
        wall_ms = (time.perf_counter() - wall0) * 1000
    return wall_ms, [r[0] * 1000 for r in results]


def imbalance(times):
    avg = sum(times) / len(times)
    return max(times) / avg if avg else 0.0


def main():
    p = mp.cpu_count()
    print("PHASE 4: SCHEDULING")
    results = {}
    wall, times = run_static(p)
    results["static"] = wall
    print(f"Contiguous static: wall={wall:.3f} ms, lambda={imbalance(times):.4f}")
    wall, times = run_cyclic(p)
    print(f"Cyclic static: wall={wall:.3f} ms, lambda={imbalance(times):.4f}")
    dynamic_results = {}
    for chunk in CHUNK_SIZES:
        wall, times = run_dynamic(p, chunk)
        dynamic_results[chunk] = wall
        print(f"Dynamic C={chunk}: wall={wall:.3f} ms, lambda={imbalance(times):.4f}")
    best_chunk = min(dynamic_results, key=dynamic_results.get)
    print(f"Best observed dynamic chunk: {best_chunk}")
    print(emit_verify_tag("PHASE4", STUDENT_ID,
                          results["static"], dynamic_results[best_chunk]))


if __name__ == "__main__":
    mp.freeze_support()
    main()
