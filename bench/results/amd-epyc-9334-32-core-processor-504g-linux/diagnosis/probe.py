"""Run benchmark.py unchanged, and note our own CPU time and page faults at every phase line it prints."""
import builtins, json, os, resource, runpy, sys, time
out = sys.argv[1]; sys.argv = ["benchmark.py"] + sys.argv[2:]
sys.path.insert(0, os.getcwd())  # import the simulator from this checkout
marks, real_print = [], builtins.print
def snapshot(label):
    r = resource.getrusage(resource.RUSAGE_SELF)
    marks.append({"label": label, "wall": time.perf_counter(), "user": r.ru_utime, "system": r.ru_stime, "minor_faults": r.ru_minflt,
                  "major_faults": r.ru_majflt, "voluntary_switches": r.ru_nvcsw, "involuntary_switches": r.ru_nivcsw})
def print_and_mark(*args, **kwargs):
    real_print(*args, **kwargs)
    text = " ".join(str(a) for a in args)
    if ": " in text and text.rstrip().endswith(" s") and len(marks) < 40:
        snapshot(text.split(":")[0])
snapshot("start"); builtins.print = print_and_mark
try:
    runpy.run_path("benchmark.py", run_name="__main__")
finally:
    json.dump(marks, open(out, "w"))
