#!/usr/bin/env python3
"""
Benchmark: torch.vmap vs a plain Python loop over rockets.

Both methods call the same `simulate_single_rocket` from vectorized_simulation.py
(forward pass only, 180 physics steps per rocket). Each (method, batch size) case
runs in a fresh subprocess so peak memory is not polluted by earlier cases.

Usage:
    python benchmark.py                       # default sizes 1 10 100 1000 10000
    python benchmark.py --sizes 1 10 100      # custom sizes
    python benchmark.py --markdown results.md # also write a markdown table
"""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time

import torch

from vectorized_simulation import device, simulate_many_rockets, simulate_single_rocket

DEFAULT_SIZES = [1, 10, 100, 1000, 10000]
METHODS = ["loop", "vmap"]


def make_inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.zeros((n, 7))
    states[:, 0] = torch.randn(n, generator=g) * 3
    states[:, 1] = 25 + torch.randn(n, generator=g) * 5
    states[:, 2] = torch.randn(n, generator=g) * 0.5
    states[:, 3] = -1.0 + torch.randn(n, generator=g) * 0.3
    states[:, 4] = torch.randn(n, generator=g) * 0.2
    states[:, 6] = 40.0
    thrusts = torch.rand(n, generator=g) * 15 + 5
    torques = torch.randn(n, generator=g) * 0.5
    return states.to(device), thrusts.to(device), torques.to(device)


def run_loop(states, thrusts, torques):
    return torch.stack(
        [simulate_single_rocket(states[i], thrusts[i], torques[i]) for i in range(len(states))]
    )


def run_vmap(states, thrusts, torques):
    return simulate_many_rockets(states, thrusts, torques)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_rss_mb():
    # ru_maxrss is KiB on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def repeats_for(method, n):
    if method == "loop" and n >= 10000:
        return 1
    if method == "loop" and n >= 1000:
        return 3
    return 5


def worker(method, n):
    """Runs one case in this process and prints a JSON result line."""
    states, thrusts, torques = make_inputs(n)
    fn = run_loop if method == "loop" else run_vmap

    with torch.no_grad():
        baseline_mb = peak_rss_mb()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        fn(states[:1], thrusts[:1], torques[:1])  # warm-up
        sync()

        times = []
        for _ in range(repeats_for(method, n)):
            start = time.perf_counter()
            out = fn(states, thrusts, torques)
            sync()
            times.append(time.perf_counter() - start)

        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_mb = peak_rss_mb() - baseline_mb

    print(json.dumps({
        "method": method,
        "n": n,
        "median_s": statistics.median(times),
        "repeats": len(times),
        "peak_mb": max(peak_mb, 0.0),
        "baseline_mb": baseline_mb,
        "checksum": out.double().sum().item(),
    }))


def run_case(method, n):
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", method, str(n)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def check_equivalence(n=64):
    states, thrusts, torques = make_inputs(n)
    with torch.no_grad():
        diff = (run_loop(states, thrusts, torques) - run_vmap(states, thrusts, torques)).abs().max()
    return diff.item()


def hardware_summary():
    if device.type == "cuda":
        hw = torch.cuda.get_device_name(0)
    else:
        hw = platform.processor() or platform.machine()
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        hw = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return {
        "device": device.type,
        "hardware": hw,
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }


def format_time(seconds):
    return f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def format_mb(mb):
    return "<0.1 MB" if mb < 0.1 else f"{mb:.1f} MB"


def to_markdown(rows, hw, max_diff):
    lines = [
        "| Rockets | Plain loop | `torch.vmap` | Speedup | Peak memory (loop) | Peak memory (vmap) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n, loop, vmap in rows:
        lines.append(
            f"| {n:,} | {format_time(loop['median_s'])} | {format_time(vmap['median_s'])} "
            f"| {loop['median_s'] / vmap['median_s']:.1f}x "
            f"| {format_mb(loop['peak_mb'])} | {format_mb(vmap['peak_mb'])} |"
        )
    if hw["device"] == "cuda":
        mem_note = "peak CUDA memory allocated"
    else:
        baseline = statistics.median(r["baseline_mb"] for _, loop, vmap in rows for r in (loop, vmap))
        mem_note = (
            "peak process RSS growth during the timed runs, on top of a "
            f"~{baseline:.0f} MB baseline for Python, PyTorch and the inputs"
        )
    lines += [
        "",
        f"Hardware: {hw['hardware']} ({hw['device'].upper()}, {hw['torch_threads']} torch threads), "
        f"PyTorch {hw['torch']}, Python {hw['python']}, {hw['os']}.",
        f"Times are medians of repeated forward runs (180 physics steps per rocket); "
        f"memory is {mem_note}. "
        f"Max difference between loop and vmap final states: {max_diff:.1e}.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--markdown", help="Write the results table to this file")
    parser.add_argument("--worker", nargs=2, metavar=("METHOD", "N"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker[0], int(args.worker[1]))
        return

    hw = hardware_summary()
    print(f"Device: {hw['device']} | {hw['hardware']} | torch {hw['torch']} | {hw['torch_threads']} threads")

    max_diff = check_equivalence()
    print(f"Loop vs vmap max abs difference: {max_diff:.1e}\n")

    rows = []
    for n in args.sizes:
        results = {}
        for method in METHODS:
            results[method] = run_case(method, n)
            r = results[method]
            print(f"{method:>4} | n={n:>6,} | {format_time(r['median_s']):>10} "
                  f"| peak {format_mb(r['peak_mb'])} | repeats {r['repeats']}")
        print(f"     speedup: {results['loop']['median_s'] / results['vmap']['median_s']:.1f}x\n")
        rows.append((n, results["loop"], results["vmap"]))

    table = to_markdown(rows, hw, max_diff)
    print(table)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(table + "\n")
        print(f"\nWrote {args.markdown}")


if __name__ == "__main__":
    main()
