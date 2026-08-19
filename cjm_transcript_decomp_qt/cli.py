"""Console-script driver for the Qt shell: the SAME argument surface,
resolution ladder, and headless hand-off as cjm-transcript-decomp-tui — all
imported from it (build_parser / resolve_split_flags / resolve_settings /
hand_off), so the two shells cannot drift on the reproducibility contract.
Only the window in the middle differs."""

import sys

from cjm_substrate_qt_kit.theme import apply_theme
from cjm_transcript_decomp_tui.cli import (build_parser, hand_off, resolve_settings,
                                           resolve_split_flags)
from PySide6.QtWidgets import QApplication

from .app import DecompWindow


def main() -> int:  # Console-script entry point (cjm-transcript-decomp-qt)
    """Resolve the shared setup surface, run the Qt batch window, hand off."""
    parser = build_parser()
    parser.prog = "cjm-transcript-decomp-qt"
    args = resolve_split_flags(parser.parse_args())
    s = resolve_settings(args)
    qapp = QApplication(sys.argv[:1])
    apply_theme(qapp)
    win = DecompWindow(s["manifests_dir"], runs_dir=s["runs_dir"],
                       sysmon_capability=s["sysmon_capability"],
                       graph_capability=args.graph_capability,
                       graph_db_path=s["graph_db_path"],
                       gap_threshold=args.gap_threshold,
                       sentence_split=args.sentence_split,
                       respine=args.respine,
                       proposals_dir=s["proposals_dir"],
                       event_split=args.event_split,
                       propset_pin=args.event_propset,
                       training_runs_dir=s["training_runs_dir"],
                       training_run_pin=s["training_run_pin"])
    win.show()
    qapp.exec()
    return hand_off(win.plan, args)
