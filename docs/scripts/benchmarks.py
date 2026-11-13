"""Tables of the documentation page "Accuracy and speed".

Run from the repository root:

    python docs/scripts/benchmarks.py

It writes docs/generated/accuracy.md and docs/generated/speed.md.  The accuracy tables compare implied vols with the 72
reference smiles in tests/data/reference_smiles.json; the timings are wall-clock times of
`eur_call_put` on the machine that runs the script.
"""
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import numpy as np

from roughprix import RoughHeston, RoughHestonParam

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "generated"
DATA = json.loads((ROOT / "tests" / "data" / "reference_smiles.json").read_text())
MATURITIES = {1 / 52: "1 week", 1 / 12: "1 month", 1.0: "1 year", 5.0: "5 years"}
SETS = ["A", "B", "C"]
PRICE_FLOOR = 1e-6    # compare vols where the out-of-the-money price is at least 1e-6 S_0
REPEATS = 7

warnings.simplefilter("ignore")


def smile(set_, H, T):
    hits = [s for s in DATA["smiles"] if s["set"] == set_ and s["H"] == H and s["T"] == T]
    assert len(hits) == 1, (set_, H, T)
    return hits[0]


def param(set_, H):
    return RoughHestonParam(**DATA["params"][set_], H=H)


def max_error_bp(iv, s):
    """Largest |vol - reference vol| in bp where the price is compared; None if a vol is missing."""
    keep = np.array(s["price"]) >= PRICE_FLOOR
    err = np.abs(np.asarray(iv, float)[keep] - np.array(s["iv"])[keep])
    return None if not np.all(np.isfinite(err)) else 1e4 * float(err.max())


def fmt_bp(x):
    if x is None:
        return "no vol"
    return f"{x:.1g}" if x < 0.01 else f"{x:.2g}" if x < 10 else f"{x:.0f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def worst(values):
    return None if any(v is None for v in values) else max(values)


def accuracy_by_base(base, Hs):
    rows = []
    for set_ in SETS:
        row = [f"{set_}"]
        for T in MATURITIES:
            errs = []
            for H in Hs:
                s = smile(set_, H, T)
                model = RoughHeston(param(set_, H), 0.0, **({} if base is None else {"base_solution": base}))
                errs.append(max_error_bp(model.eur_call_put(np.array(s["K"]), T, implied_vol=True), s))
            row.append(fmt_bp(worst(errs)))
        rows.append(row)
    return rows


def accuracy_expansion(cases, order):
    rows = []
    for anchor, targets in cases:
        for H in targets:
            row = [f"{anchor:g} → {H:g}"]
            for T in MATURITIES:
                errs = []
                for set_ in SETS:
                    s = smile(set_, H, T)
                    model = RoughHeston(param(set_, anchor), 0.0)
                    iv = model.eur_call_put(np.array(s["K"]), T, implied_vol=True, H=H, expansion_order=order)
                    errs.append(max_error_bp(iv, s))
                row.append(fmt_bp(worst(errs)))
            rows.append(row)
    return rows


def median_ms(call):
    call()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def machine():
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    else:
        cpu = platform.processor() or platform.machine()
    return (f"{cpu}, {os.cpu_count()} cores; Python {platform.python_version()}, JAX {jax.__version__}, "
            f"NumPy {np.__version__}, roughprix {_version()}")


def _version():
    import roughprix
    if Path(roughprix.__file__).resolve().is_relative_to(ROOT / "src"):   # the source tree
        return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    try:
        return version("roughprix")
    except PackageNotFoundError:
        return "(unknown version)"


def main():
    started = time.perf_counter()
    load = [os.getloadavg()[0]]
    out = []

    # First call in a fresh process: includes the JIT compilation.
    s = smile("A", 0.1, 1.0)
    t0 = time.perf_counter()
    RoughHeston(param("A", 0.1), 0.0).eur_call_put(np.array(s["K"]), 1.0)
    first_call = time.perf_counter() - t0

    ref_err = max(1e4 * max(np.maximum(s["err_time"], s["err_fourier"])[np.array(s["price"]) >= PRICE_FLOOR])
                  for s in DATA["smiles"])
    out.append(f"The reference's own error estimates stay below {ref_err:.2g} bp at every compared strike.\n")
    cols = ["Set"] + list(MATURITIES.values())

    out.append("### Default settings\n")
    out.append(table(cols, accuracy_by_base(None, [0.5, 0.3, 0.1, 0.0, -0.2, -0.4])))
    out.append("\n### Root-Padé base\n")
    out.append(table(cols, accuracy_by_base("root_pade", [0.3, 0.1, 0.0, -0.2, -0.4])))
    for order in (2, 4):
        out.append(f"\n### H-expansion of order {order}\n")
        out.append(table(["$H_0 \\to H$"] + list(MATURITIES.values()),
                         accuracy_expansion([(0.0, [0.1, -0.2]), (0.5, [0.3])], order)))

    # Speed: nine strikes of set A, default settings unless stated.
    timing_rows = []
    configs = [
        ("Default, $H = 0.1$", lambda T, K: RoughHeston(param("A", 0.1), 0.0), {}),
        ("Root-Padé base, $H = 0.1$", lambda T, K: RoughHeston(param("A", 0.1), 0.0, base_solution="root_pade"), {}),
        ("Exact Heston, $H = \\tfrac12$", lambda T, K: RoughHeston(param("A", 0.5), 0.0), {}),
        ("Expansion $N = 4$, $H_0 = 0 \\to H = 0.1$", lambda T, K: RoughHeston(param("A", 0.0), 0.0),
         dict(H=0.1, expansion_order=4)),
    ]
    for label, make, kw in configs:
        row = [label]
        for T in MATURITIES:
            K = np.array(smile("A", 0.1, T)["K"])
            model = make(T, K)
            row.append(f"{median_ms(lambda: model.eur_call_put(K, T, **kw)):.0f}")
        timing_rows.append(row)
    acc, out = out, []
    load.append(os.getloadavg()[0])
    out.append("### Time per call\n")
    out.append(table(["Nine strikes, set A"] + list(MATURITIES.values()), timing_rows))

    many_rows = []
    H_targets = np.linspace(-0.2, 0.2, 21)
    for T in (1 / 12, 1.0):
        K = np.array(smile("A", 0.1, T)["K"])
        expansion = RoughHeston(param("A", 0.0), 0.0)
        t_exp = median_ms(lambda: expansion.eur_call_put(K, T, H=H_targets, expansion_order=4))
        direct = [RoughHeston(param("A", float(H)), 0.0) for H in H_targets]
        t_dir = sum(median_ms(lambda m=m: m.eur_call_put(K, T)) for m in direct)
        many_rows.append([MATURITIES[T], f"{t_exp:.0f}", f"{t_dir:.0f}"])
    out.append("\n### Many values of H in one call\n")
    out.append(table(["Nine strikes, 21 values of $H$ in $[-0.2, 0.2]$", "Expansion $N = 4$ from $H_0 = 0$, one call",
                      "Default solver, one call per $H$"], many_rows))

    out.append(f"\nTimes in milliseconds, medians of {REPEATS} calls after a first call. The first call in a "
               f"session compiles; here it took {first_call:.1f} s. Machine: {machine()}. Load average during "
               f"the timings: {np.mean(load + [os.getloadavg()[0]]):.1f}.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "accuracy.md").write_text("\n".join(acc) + "\n", encoding="utf-8")
    (OUT / "speed.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}/accuracy.md and speed.md in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
