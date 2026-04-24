#!/usr/bin/env python3
"""Write AXPI010 pattern file for verilog_eval_* and axsim_main --patterns-file."""

from __future__ import annotations

import argparse
import pathlib
import sys

_LIB = pathlib.Path(__file__).resolve().parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from pattern_io import write_axpi010


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate AXPI010 shared PI pattern file")
    ap.add_argument("output", type=pathlib.Path, help="Output .axpi path")
    ap.add_argument("--bits", type=int, required=True, help="Total PI bit width (must match circuit num_pis)")
    ap.add_argument("--patterns", type=int, required=True, help="Number of patterns")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_axpi010(args.output, args.seed, args.bits, args.patterns)
    print(f"Wrote {args.output} (bits={args.bits} patterns={args.patterns} seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
