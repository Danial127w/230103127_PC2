import multiprocessing as mp,math
from harness import run_benchmark,emit_verify_tag
STUDENT_ID=230103127; NS=1_200_000; N0=200_000
def stage_a(n):
    z=0.
    for i in range(n):
        x=(i%997)*.001+.1
        for _ in range(5): z+=math.sin(x)+math.sqrt(x+1); x+=.00001
    return z
def stage_b(bounds):
    lo,hi=bounds; z=0.
    for i in range(lo,hi):
        x=(i%997)*.001+.1
        for _ in range(15): x=x*1.0000001+.0000001; z+=math.sin(x)*math.sqrt(x+1)
    return z
def pipe(pool,n,p):
    stage_a(max(1,int(.05*n))); c=(n+p-1)//p
    pool.map(stage_b,[(i*c,min(n,(i+1)*c)) for i in range(p)])
def main():
    P=mp.cpu_count(); strong={}; ta1=0
    print("PHASE 2: STRONG / WEAK SCALING")
    for p in range(1,P+1):
        with mp.Pool(p) as pool:
            ta=run_benchmark(lambda:stage_a(int(.05*NS)),1,3).median_ms
            st=run_benchmark(lambda:pipe(pool,NS,p),1,3)
        strong[p]=st.median_ms
        if p==1: t1=st.median_ms; ta1=ta
        sp=t1/st.median_ms
        print(f"STRONG p={p} Tp={st.median_ms:.3f} ms S={sp:.3f} E={sp/p:.3f} StageA={ta:.3f} ms")
    s=ta1/t1; print(f"Amdahl serial fraction={s:.5f}; cap={1/s:.3f}")
    weak={}; gust={}
    for p in range(1,P+1):
        n=p*N0
        with mp.Pool(p) as pool:
            ta=run_benchmark(lambda:stage_a(int(.05*n)),1,3).median_ms
            st=run_benchmark(lambda:pipe(pool,n,p),1,3)
        weak[p]=st.median_ms; alpha=ta/st.median_ms
        ew=weak[1]/st.median_ms if p>1 else 1.; g=p-alpha*(p-1); gust[p]=g
        print(f"WEAK p={p} N={n} Tp={st.median_ms:.3f} ms Eweak={ew:.3f} alpha={alpha:.4f} Gustafson={g:.3f}")
    print(emit_verify_tag("PHASE2",STUDENT_ID,t1/strong[P],gust[P]))
if __name__=="__main__": main()
