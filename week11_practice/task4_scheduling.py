import math,multiprocessing as mp,time
from harness import emit_verify_tag
STUDENT_ID=230103127; N=8000
def item(i):
    k=1+int((i/40)**2); x=i*.0001+.1; z=0.
    for _ in range(k): x+=.000001; z+=math.sin(x)+math.cos(x)+math.sqrt(x)
    return z
def block_worker(w,lo,hi,res,times):
    t=time.perf_counter(); s=sum(item(i) for i in range(lo,hi)); times[w]=time.perf_counter()-t; res[w]=s
def cyclic_worker(w,p,res,times):
    t=time.perf_counter(); s=sum(item(i) for i in range(w,N,p)); times[w]=time.perf_counter()-t; res[w]=s
def dynamic_worker(w,counter,lock,chunk,res,times):
    t=time.perf_counter(); s=0.
    while True:
        with lock: lo=counter.value; counter.value+=chunk
        if lo>=N: break
        for i in range(lo,min(N,lo+chunk)): s+=item(i)
    times[w]=time.perf_counter()-t; res[w]=s
def run(kind,p,chunk=1):
    ctx=mp.get_context(); res=ctx.Array('d',p,lock=False); times=ctx.Array('d',p,lock=False); counter=ctx.Value('i',0); lock=ctx.Lock(); ps=[]; t=time.perf_counter()
    for w in range(p):
        if kind=="block": args=(w,w*N//p,(w+1)*N//p,res,times); target=block_worker
        elif kind=="cyclic": args=(w,p,res,times); target=cyclic_worker
        else: args=(w,counter,lock,chunk,res,times); target=dynamic_worker
        q=ctx.Process(target=target,args=args); q.start(); ps.append(q)
    for q in ps:
        q.join()
        if q.exitcode: raise RuntimeError("worker failed")
    wall=(time.perf_counter()-t)*1000; vals=list(times); avg=sum(vals)/p
    return wall,max(vals)/avg if avg else 0.
def main():
    p=min(8,mp.cpu_count()); print("PHASE 4: SCHEDULING")
    b,lb=run("block",p); c,lc=run("cyclic",p)
    print(f"Contiguous static: wall={b:.3f} ms lambda={lb:.4f}"); print(f"Cyclic static: wall={c:.3f} ms lambda={lc:.4f}")
    d={}
    for ch in (1,8,64,512):
        w,l=run("dynamic",p,ch); d[ch]=(w,l); print(f"Dynamic C={ch}: wall={w:.3f} ms lambda={l:.4f}")
    best=min(d,key=lambda x:d[x][0]); print("Best observed dynamic chunk:",best)
    print(emit_verify_tag("PHASE4",STUDENT_ID,b,d[best][0]))
if __name__=="__main__": main()
