import multiprocessing as mp,time
from harness import emit_verify_tag
STUDENT_ID=230103127
def main():
    n=1_000_000; p=mp.cpu_count(); t=time.perf_counter_ns()
    for _ in range(n): time.perf_counter_ns()
    avg=(time.perf_counter_ns()-t)/n
    print("PHASE 0: HARDWARE TELEMETRY"); print("Logical CPUs:",p)
    print(f"Average timer latency: {avg:.3f} ns")
    print(emit_verify_tag("PHASE0",STUDENT_ID,avg,float(p)))
if __name__=="__main__": main()
