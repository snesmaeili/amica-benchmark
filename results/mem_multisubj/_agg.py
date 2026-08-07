"""Aggregate multi-subject CPU peak-RSS across ds004505 sub-01..06 for the paper note."""
import glob
import json
import os
import statistics
from collections import defaultdict

dirs = ["results/mem_compare/cpu/ds004505_sub-01_mem"] + sorted(
    glob.glob("results/mem_multisubj/ds004505_sub-*_mem")
)
by_impl = defaultdict(list)  # impl -> [(sub, rss, delta, T)]
for d in dirs:
    sub = os.path.basename(d).split("_")[1]
    for jf in glob.glob(os.path.join(d, "*.json")):
        a = json.load(open(jf))
        impl = a.get("implementation")
        rss = a.get("peak_rss_gb")
        if impl and rss:
            by_impl[impl].append((sub, rss, a.get("delta_rss_gb"), a.get("n_samples")))

print(f"{'impl':32s} {'n':>2s} {'med':>6s} {'min':>6s} {'max':>6s}  (peak RSS GB)")
for impl in sorted(by_impl):
    vals = [r for _, r, _, _ in by_impl[impl]]
    print(f"{impl:32s} {len(vals):2d} {statistics.median(vals):6.2f} "
          f"{min(vals):6.2f} {max(vals):6.2f}")
    for sub, r, dl, ns in sorted(by_impl[impl]):
        dls = f"{dl:.2f}" if dl is not None else "  -"
        print(f"    {sub}: peak={r:6.2f}  delta={dls:>5s}  T={ns}")
