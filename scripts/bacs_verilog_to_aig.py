#!/usr/bin/env python3
"""
Convert BACS Verilog benchmarks to AIG (.aig) using Yosys.

Scans each subdirectory of BACS for *.v, skips testbenches (*_test.v), and writes
<stem>.aig next to each source file. Top module is inferred with hierarchy -auto-top
(so names like classifier.v -> module svm still work).

Combinational (or fully synth'd to AND/NOT) designs work out of the box. Purely
combinational BACS benches convert successfully; sequential benchmarks (fft,
classifier) still contain flip-flops after synth — Yosys write_aiger then fails
(unsupported $_DFF_/$_SDFF_*). Those need a custom flow (e.g. unrolling, or a
toolchain that emits sequential AIGER) and are reported as FAIL.

Requires: yosys on PATH (https://yosyshq.net/yosys/).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
from typing import List, Tuple


def iter_verilog(bacs_root: pathlib.Path) -> List[pathlib.Path]:
    out: List[pathlib.Path] = []
    for p in sorted(bacs_root.rglob("*.v")):
        if ".git" in p.parts:
            continue
        if p.name.endswith("_test.v"):
            continue
        out.append(p)
    return out


def yosys_script(verilog: pathlib.Path, aig_out: pathlib.Path, use_synth: bool) -> str:
    v = verilog.as_posix()
    o = aig_out.as_posix()
    lines = [
        f"read_verilog {v}",
        "hierarchy -auto-top",
    ]
    # AIGER writer only accepts AND/NOT-style mapping; AND,OR leaves $_OR_ and breaks write_aiger.
    if use_synth:
        lines += ["synth -flatten", "abc -g AND"]
    else:
        lines += [
            "proc",
            "opt",
            "flatten",
            "opt",
            "techmap",
            "opt",
            "abc -g AND",
        ]
    lines.append(f"write_aiger -symbols {o}")
    return "\n".join(lines) + "\n"


def run_yosys(ys: str, yosys_bin: str) -> Tuple[int, str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ys",
        delete=False,
        encoding="utf-8",
    ) as tf:
        tf.write(ys)
        ys_path = tf.name
    try:
        cp = subprocess.run(
            [yosys_bin, "-s", ys_path],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        msg = (cp.stdout or "") + (cp.stderr or "")
        return cp.returncode, msg
    finally:
        pathlib.Path(ys_path).unlink(missing_ok=True)


def convert_one(
    verilog: pathlib.Path,
    yosys_bin: str,
    force: bool,
) -> Tuple[str, str]:
    aig = verilog.with_suffix(".aig")
    if aig.exists() and not force:
        return "SKIP", "exists"

    try:
        if verilog.stat().st_size < 16:
            return "SKIP", "empty or trivial file"
    except OSError as e:
        return "FAIL", str(e)

    for use_synth in (True, False):
        ys = yosys_script(verilog, aig, use_synth=use_synth)
        code, log = run_yosys(ys, yosys_bin)
        if code == 0 and aig.is_file():
            return "OK", ("synth" if use_synth else "flatten")
        aig.unlink(missing_ok=True)
        last_log = log
    return "FAIL", last_log[-8000:] if last_log else "no output"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert BACS Verilog files to AIG via Yosys")
    ap.add_argument(
        "--bacs-root",
        type=pathlib.Path,
        default=None,
        help="BACS directory (default: <repo>/BACS next to this script)",
    )
    ap.add_argument("--yosys", default="yosys", help="Yosys executable")
    ap.add_argument("--force", action="store_true", help="Overwrite existing .aig")
    ap.add_argument("--dry-run", action="store_true", help="List files only")
    args = ap.parse_args()

    root = args.bacs_root
    if root is None:
        root = pathlib.Path(__file__).resolve().parent.parent / "BACS"
    root = root.resolve()
    if not root.is_dir():
        print(f"ERROR: BACS root not found: {root}", file=sys.stderr)
        return 1

    files = iter_verilog(root)
    if not files:
        print(f"No .v files under {root}", file=sys.stderr)
        return 1

    print(f"BACS root: {root}")
    print(f"Found {len(files)} Verilog file(s) (excluding *_test.v)")

    if args.dry_run:
        for p in files:
            print(p)
        return 0

    ok = skip = fail = 0
    for p in files:
        status, detail = convert_one(p, args.yosys, args.force)
        rel = p.relative_to(root)
        if status == "OK":
            print(f"OK   {rel}  ({detail})")
            ok += 1
        elif status == "SKIP":
            print(f"SKIP {rel}  ({detail})")
            skip += 1
        else:
            print(f"FAIL {rel}", file=sys.stderr)
            print(detail, file=sys.stderr)
            fail += 1

    print(f"Done: OK={ok} SKIP={skip} FAIL={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
