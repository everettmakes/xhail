"""
xhail.cli — command-line interface
====================================

Installed as the ``xhail`` console script via ``pyproject.toml``.

Usage examples::

    xhail run penguins.lp
    xhail run penguins.lp --depth 15 --verbose
    xhail run penguins.lp --debug --debug-output ./debug_output/
    xhail --version
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .core import learn
from .parser.parser import ParseError
from .reasoning.deduction import DeductionTimeoutError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xhail",
        description=(
            "XHAIL: eXtended Hybrid Abductive Inductive Learning.\n\n"
            "Learns logic-program rules from background knowledge and examples\n"
            "using a three-phase abduction → deduction → induction pipeline\n"
            "built on Answer Set Programming (clingo)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"xhail {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ------------------------------------------------------------------ run
    run_p = sub.add_parser(
        "run",
        help="Run the XHAIL learner on an input .lp file.",
        description="Run the XHAIL learner on an input .lp file and print the learned hypothesis.",
    )
    run_p.add_argument(
        "file",
        type=Path,
        metavar="FILE",
        help="Path to the input .lp file (background knowledge + examples + mode declarations).",
    )
    run_p.add_argument(
        "--depth",
        type=int,
        default=10,
        metavar="N",
        help="Maximum deduction depth (default: 10).",
    )
    run_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print phase progress to stderr.",
    )
    run_p.add_argument(
        "--debug",
        action="store_true",
        help="Write intermediate ASP programs to --debug-output.",
    )
    run_p.add_argument(
        "--debug-output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for intermediate ASP files when --debug is set (default: ./xhail_debug/).",
    )
    run_p.add_argument(
        "--all",
        "-a",
        action="store_true",
        dest="all_solutions",
        help="Enumerate and print all optimal hypotheses, not just the first.",
    )
    run_p.add_argument(
        "--kill",
        "-k",
        type=int,
        default=None,
        metavar="N",
        help="Stop solving after N seconds (timeout).",
    )
    run_p.add_argument(
        "--terminate",
        "-t",
        action="store_true",
        help="Stop after finding the first hypothesis (default behaviour; explicit flag for clarity).",
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``xhail`` console script.

    Returns:
        Exit code (0 = success, 1 = user error, 2 = runtime error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    if args.command == "run":
        return _cmd_run(args)

    parser.print_help()  # pragma: no cover — unreachable with sub.required=True
    return 1  # pragma: no cover


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        result = learn(
            args.file,
            depth=args.depth,
            debug=args.debug,
            debug_output_dir=args.debug_output,
            all_solutions=args.all_solutions,
            timeout=args.kill,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ParseError as exc:
        print(f"error: parse failed — {exc}", file=sys.stderr)
        return 1
    except DeductionTimeoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if result.success:
        if args.all_solutions and len(result.all_hypotheses) > 1:
            for idx, hyp in enumerate(result.all_hypotheses, 1):
                print(f"% Hypothesis {idx}:")
                for rule in hyp:
                    print(rule)
                if idx < len(result.all_hypotheses):
                    print()
        else:
            for rule in result.hypothesis:
                print(rule)
    else:
        print("No hypothesis found.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
