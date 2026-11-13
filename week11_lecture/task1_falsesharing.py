import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes

from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103127
M = 24_000_000
STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    try:
        acc = arr[slot_idx]
        mask = 0xFFFFFFFFFFFFFFFF
        mult = 6364136223846793005
        for i in range(iters):
            acc = (acc * mult + (i | 1)) & mask
            arr[slot_idx] = acc
    finally:
        del arr
        shm.close()


def run_test(p, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        for t in range(p):
            arr[t * stride] = 0
        del arr
        processes = [
            mp.Process(target=fs_worker,
                       args=(shm.name, t * stride, M // p))
            for t in range(p)
        ]
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
        bad = [proc.exitcode for proc in processes if proc.exitcode != 0]
        if bad:
            raise RuntimeError(f"Worker exit codes: {bad}")
    finally:
        shm.close()
        shm.unlink()


def benchmark_test(p, stride):
    return run_benchmark(lambda: run_test(p, stride),
                         warmup_runs=2, timed_runs=7)


def calculate_metrics(t1, tp, p):
    speedup = t1 / tp
    efficiency = speedup / p
    karp_flatt = 0.0 if p == 1 else (
        (1 / speedup) - (1 / p)
    ) / (1 - (1 / p))
    return speedup, efficiency, karp_flatt


def main():
    physical_cores = mp.cpu_count()
    stride_results = {}
    print("PHASE 1: FALSE SHARING")
    print("TABLE 1A: stride, bytes, median_ms, IQR_ms, Mops/s")
    for stride in STRIDES:
        stats = benchmark_test(physical_cores, stride)
        throughput = M / (stats.median_ms * 1000)
        stride_results[stride] = stats.median_ms
        print(f"{stride}, {stride * 8}, {stats.median_ms:.3f}, "
              f"{stats.iqr_ms:.3f}, {throughput:.3f}")

    print("\nTABLE 1B")
    for stride in (1, 16):
        timings = {}
        for p in PROCESS_COUNTS:
            timings[p] = benchmark_test(p, stride).median_ms
        t1 = timings[1]
        print(f"\nstride={stride}: p,Tp_ms,speedup,efficiency,e(p)")
        for p in PROCESS_COUNTS:
            s, e, kf = calculate_metrics(t1, timings[p], p)
            print(f"{p},{timings[p]:.3f},{s:.4f},{e:.4f},{kf:.5f}")

    print(emit_verify_tag("PHASE1", STUDENT_ID,
                          stride_results[1], stride_results[16]))


if __name__ == "__main__":
    mp.freeze_support()
    main()
