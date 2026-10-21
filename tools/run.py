#!/usr/bin/env python3
"""Run a program: on Spike, or on this machine (SOC=host).

usage: tools/run.py SOC ELF [--cycles N] [--expect TEXT]
  SOC = spike:MARCH/MABI   Spike (SPIKE: its path, default spike on PATH)
      | host               the program built for this machine (SOC=host): run directly
A program built for a target directory runs on that SoC through the tool that describes it (its RTL simulation or
its board); this script says so and stops.
Prints the console; exit status: the program's exit code (125 when it did not exit, e.g. the cycle limit).
"""
import argparse
import os
import re
import subprocess
import sys


def run_spike(spec, elf, cycles):
    march = spec[6:].split("/")[0]
    spike = os.environ.get("SPIKE", "spike")
    r = subprocess.run([spike, f"--isa={march}", "-m0x80000000:0x10000000", f"--instructions={cycles}", elf],
                       capture_output=True, text=True, timeout=3600)
    return r.stdout + r.stderr


def run_host(elf):
    r = subprocess.run([elf], capture_output=True, text=True, timeout=3600)
    return r.stdout + r.stderr + f"\n[exit {r.returncode}]\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("soc")
    ap.add_argument("elf")
    ap.add_argument("--cycles", type=int, default=50_000_000)
    ap.add_argument("--expect", help="text the console must contain")
    a = ap.parse_args(argv)
    if not (a.soc.startswith("spike:") or a.soc == "host"):
        print(f"run.py: {a.elf} is built for the SoC described by {a.soc}: run it with that SoC's tool "
              f"(its RTL simulation or board)", file=sys.stderr)
        return 2
    out = run_spike(a.soc, a.elf, a.cycles) if a.soc.startswith("spike:") else run_host(a.elf)
    print(out, end="" if out.endswith("\n") else "\n")
    m = re.search(r"\[exit (-?\d+)\]", out)
    code = int(m.group(1)) if m else 125
    if a.expect and a.expect not in out:
        print(f"run.py: '{a.expect}' not in the console", file=sys.stderr)
        return code or 1
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
