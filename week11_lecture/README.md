# ParallelComputingLab — Python project

Open this folder in PyCharm. Standard library only; no pip packages required.

## Run order
```bash
python3 task0_cal.py
python3 task1_falsesharing.py
python3 task2_scaling.py
python3 task3a_roofline.py
python3 task3b_gemm.py
python3 task4_scheduling.py
```

Run the heavier phases individually and allow them to finish. Benchmarks can take a long time on a laptop. Record actual output in `lab_report_230103127.md`.

Important implementation notes:
- Phase 1 follows the shared-memory stride/process sweep.
- Phase 2 uses a persistent pool during timed pipeline calls, but the weak-scaling pool is also created outside each timed call.
- Phase 3A includes pool creation in the measured copy/compute wrapper; disclose this if comparing results.
- Phase 3B is pure Python and can be slow for N=384, especially naive GEMM.
- Results and hashes depend on the actual run and environment.
