from dataclasses import dataclass
import multiprocessing as mp, time
@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
def fnv1a_hash(s):
    h=14695981039346656037
    for b in s.encode("utf-8"): h=((h^b)*1099511628211)&0xffffffffffffffff
    return h
def run_benchmark(fn,warmup=2,runs=7):
    for _ in range(warmup): fn()
    a=[]
    for _ in range(runs):
        t=time.perf_counter_ns(); fn(); a.append((time.perf_counter_ns()-t)/1e6)
    a.sort()
    return BenchStats(a[0],a[len(a)//2],a[3*runs//4]-a[runs//4])
def emit_verify_tag(phase,seed,m1,m2):
    p=mp.cpu_count(); h=fnv1a_hash(f"{phase}|{seed}|{p}|{m1:.4f}|{m2:.4f}")
    return f"VERIFY_HASH[{phase}]: {h:016X} (P_log={p})"
