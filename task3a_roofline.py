
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes

from harness import run_benchmark

STUDENT_ID = 230103127

BUFFER_BYTES = 96 * 1_000_000
FLOPS_PER_WORKER = 80_000_000


def _copy_worker(args):
    name_a, name_b, start, end = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    try:
        shm_a.buf[start:end] = shm_b.buf[start:end]
    finally:
        shm_a.close()
        shm_b.close()


def _compute_worker(flops):
    x = [0.1 + i * 0.001 for i in range(16)]

    a = 1.0000001
    b = 0.0000003

    iterations = flops // 32

    for _ in range(iterations):
        for j in range(16):
            x[j] = x[j] * a + b

    return sum(x)


def main():
    p = mp.cpu_count()

    print("=" * 55)
    print("PHASE 3A: EMPIRICAL ROOFLINE")
    print("=" * 55)

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_BYTES
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_BYTES
    )

    try:
        # Initialize shared memory using ctypes
        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(shm_a.buf)),
            0,
            BUFFER_BYTES
        )

        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(shm_b.buf)),
            1,
            BUFFER_BYTES
        )

        # Prepare parallel copy tasks
        chunk = BUFFER_BYTES // p

        tasks = [
            (
                shm_a.name,
                shm_b.name,
                i * chunk,
                BUFFER_BYTES if i == p - 1
                else (i + 1) * chunk
            )
            for i in range(p)
        ]

        def copy_all():
            with mp.Pool(processes=p) as pool:
                pool.map(_copy_worker, tasks)

        print("\n1. DRAM BANDWIDTH")

        bw_stats = run_benchmark(
            copy_all,
            warmup_runs=1,
            timed_runs=3
        )

        bandwidth = (
            2 * BUFFER_BYTES
        ) / (bw_stats.median_ms * 1_000_000)

        print(f"Median time: {bw_stats.median_ms:.3f} ms")
        print(f"Bandwidth: {bandwidth:.4f} GB/s")

        print("\n2. PYTHON COMPUTE CEILING")

        def compute_all():
            with mp.Pool(processes=p) as pool:
                pool.map(
                    _compute_worker,
                    [FLOPS_PER_WORKER] * p
                )

        comp_stats = run_benchmark(
            compute_all,
            warmup_runs=1,
            timed_runs=3
        )

        total_flops = FLOPS_PER_WORKER * p

        compute_gflops = (
            total_flops / (comp_stats.median_ms * 1_000_000)
        )

        print(f"Median time: {comp_stats.median_ms:.3f} ms")
        print(f"Compute: {compute_gflops:.4f} GFLOP/s")

        print("\n3. EMPIRICAL RIDGE POINT")

        ridge = compute_gflops / bandwidth

        print(f"Ridge Point: {ridge:.4f} FLOPs/Byte")

        print("\nPHASE 3A COMPLETED")

    finally:
        shm_a.close()
        shm_a.unlink()

        shm_b.close()
        shm_b.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
