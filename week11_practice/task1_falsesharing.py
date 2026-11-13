import ctypes,multiprocessing as mp,time
from multiprocessing import shared_memory
from harness import emit_verify_tag
STUDENT_ID=230103127; M=16_000_000; MASK=0xffffffffffffffff
def worker(name,slot,iters):
    s=shared_memory.SharedMemory(name=name); a=(ctypes.c_uint64*512).from_buffer(s.buf); x=a[slot]
    for i in range(iters):
        x=(x*6364136223846793005+(i|1))&MASK; a[slot]=x
    del a; s.close()
def run(p,stride):
    s=shared_memory.SharedMemory(create=True,size=4096)
    try:
        a=(ctypes.c_uint64*512).from_buffer(s.buf)
        for i in range(512): a[i]=i+1
        del a
        ps=[]; t=time.perf_counter_ns()
        for w in range(p):
            q=mp.Process(target=worker,args=(s.name,w*stride,M//p)); q.start(); ps.append(q)
        for q in ps:
            q.join()
            if q.exitcode: raise RuntimeError("worker failed")
        return (time.perf_counter_ns()-t)/1e6
    finally: s.close(); s.unlink()
def med(p,st):
    x=sorted(run(p,st) for _ in range(3)); return x[1],x[2]-x[0]
def main():
    p=mp.cpu_count(); print("PHASE 1: FALSE SHARING")
    vals={}
    for st in (1,2,4,8,16,32):
        m,iq=med(p,st); vals[st]=m
        print(f"stride={st} bytes={st*8} median={m:.3f} ms spread={iq:.3f} throughput={M/(m*1000):.3f} Mops/s")
    for st in (1,16):
        t1=None
        for n in sorted(set((1,2,4,8,p))):
            m,_=med(n,st); t1=m if n==1 else t1; sp=t1/m; e=sp/n
            k=(1/sp-1/n)/(1-1/n) if n>1 else 0
            print(f"stride={st} p={n} Tp={m:.3f} ms S={sp:.3f} E={e:.3f} Karp-Flatt={k:.5f}")
    print(emit_verify_tag("PHASE1",STUDENT_ID,vals[1],vals[16]))
if __name__=="__main__": main()
