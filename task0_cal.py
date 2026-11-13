import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230103127


def main():
    print("Logical CPUs:", mp.cpu_count())
    reads = 2_000_000
    start = time.perf_counter_ns()
    for _ in range(reads):
        time.perf_counter_ns()
    elapsed = time.perf_counter_ns() - start
    average_ns = elapsed / reads
    print(f"Average timer latency: {average_ns:.5f} ns")
    print(emit_verify_tag("PHASE0", STUDENT_ID,
                          average_ns, float(mp.cpu_count())))


if __name__ == "__main__":
    mp.freeze_support()
    main()
