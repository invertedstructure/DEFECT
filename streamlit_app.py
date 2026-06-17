"""
Streamlit app for the F₂ suspension‑defect experiment.

This application provides a simple user interface to run the finite (F₂)
simplicial suspension experiment without requiring any familiarity with
Python. Users may select a range of cycle lengths, a maximum suspension
depth, and optionally enable a random relabel anti‑cheat check. Upon
running, the app displays summary statistics and a table of results,
together with diagnostic information if any failures occur.

The core mathematics is delegated entirely to the ``defect_core`` module.
No shortcuts or parity inferences are used; each lifted cochain is
explicitly tested against the coboundary matrix of the lifted complex.

Instructions for running the app are provided in the accompanying
``README.md``.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from defect_core import iter_experiment
import time

# Global flag: disable or enable disk checkpointing. When False, no
# filesystem writes are attempted. When True, the app will try
# to write export files into the ``exports`` directory and silently
# ignore permission errors.
ENABLE_DISK_CHECKPOINT = False

# ----------------------------------------------------------------------------
# Utility functions for summarising and exporting experiment results
# ----------------------------------------------------------------------------

def build_trajectory_summary(rows: list[dict]) -> list[dict]:
    """Compute a trajectory summary for each cycle_n based on a list of rows.

    Each row is assumed to contain at least the keys 'cycle_n',
    'suspension_depth' and 'status_symbol'. The summary captures the
    trajectory of trivial/nontrivial/error statuses across depths for each
    cycle length.

    Args:
        rows: list of result dictionaries from ``iter_experiment``.

    Returns:
        A list of dictionaries, one per distinct cycle_n, containing:
            cycle_n (int)
            seed_status_symbol (str)
            depths_completed (str)
            trajectory_string (str)
            flip_count (int)
            flip_depths (list[int])
            longest_T_run (int)
            longest_N_run (int)
            terminal_status (str)
            strict_preserved_all_completed (bool)
            first_failure_depth (int or None)
    """
    # Group rows by cycle_n
    grouped: dict[int, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r.get("cycle_n"), []).append(r)
    summaries: list[dict] = []
    for n_val, group in grouped.items():
        # Sort by suspension_depth to build the trajectory in order
        group_sorted = sorted(group, key=lambda r: r.get("suspension_depth", 0))
        statuses = [r.get("status_symbol", 'E') for r in group_sorted]
        depths = [r.get("suspension_depth", 0) for r in group_sorted]
        if depths:
            depth_range = f"{min(depths)}..{max(depths)}"
        else:
            depth_range = ""
        trajectory_string = "".join(statuses)
        # Seed status is the first status symbol
        seed_status = statuses[0] if statuses else ''
        # Count flips and record depths at which flips occur (status differs from seed)
        flip_count = 0
        flip_depths: list[int] = []
        for idx, s in enumerate(statuses):
            if idx == 0:
                continue
            if s != seed_status:
                flip_count += 1
                flip_depths.append(depths[idx])
        # Compute longest runs of T and N
        longest_T = 0
        longest_N = 0
        current_T = 0
        current_N = 0
        for s in statuses:
            if s == 'T':
                current_T += 1
                longest_T = max(longest_T, current_T)
                current_N = 0
            elif s == 'N':
                current_N += 1
                longest_N = max(longest_N, current_N)
                current_T = 0
            else:
                # Reset counts on error
                current_T = 0
                current_N = 0
        terminal_status = statuses[-1] if statuses else ''
        # Strictly preserved so far if all statuses equal seed and no errors
        strict_preserved = all((s == seed_status and s != 'E') for s in statuses)
        # First failure depth is first depth where preserved is False or error exists
        first_failure_depth = None
        for r in group_sorted:
            if not r.get("preserved", True) or r.get("error"):
                first_failure_depth = r.get("suspension_depth")
                break
        summaries.append({
            "cycle_n": n_val,
            "seed_status_symbol": seed_status,
            "depths_completed": depth_range,
            "trajectory_string": trajectory_string,
            "flip_count": flip_count,
            "flip_depths": flip_depths,
            "longest_T_run": longest_T,
            "longest_N_run": longest_N,
            "terminal_status": terminal_status,
            "strict_preserved_all_completed": strict_preserved,
            "first_failure_depth": first_failure_depth,
        })
    # Sort by cycle_n for deterministic ordering
    return sorted(summaries, key=lambda d: d.get("cycle_n"))


def make_rows_csv(rows: list[dict]) -> str:
    """Return a CSV representation of rows as a string.

    Args:
        rows: list of result dictionaries

    Returns:
        A CSV string containing all rows.
    """
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def make_rows_jsonl(rows: list[dict]) -> str:
    """Return a JSON Lines representation of rows.

    Args:
        rows: list of result dictionaries

    Returns:
        A string with one JSON object per line.
    """
    return "\n".join(json.dumps(r) for r in rows)


def make_trajectories_txt(rows: list[dict]) -> str:
    """Return a plain‑text representation of trajectory summaries.

    Args:
        rows: list of result dictionaries

    Returns:
        A string where each line contains 'cycle_n: trajectory_string' for
        each cycle length present in ``rows``.
    """
    traj_lines = []
    for ts in build_trajectory_summary(rows):
        traj_lines.append(f"{ts['cycle_n']}: {ts['trajectory_string']}")
    return "\n".join(traj_lines)


def render_markdown_receipt(rows: list[dict], config: dict) -> str:
    """Render a human‑readable Markdown receipt from result rows and config.

    The receipt summarises the run configuration, completion statistics,
    trajectory summary, failure rows, runtime profile, the last completed
    row, and the row schema. It is deterministic and safe for copying
    into another document or chat.

    Args:
        rows: list of result dictionaries from ``iter_experiment``.
        config: dictionary containing run configuration parameters. The keys
            expected include n_min, n_max, max_depth, do_random_relabel,
            relabel_trials, stop_after_failure, total_cases, run_mode.

    Returns:
        A Markdown string representing the experiment receipt.
    """
    # Prepare summary statistics
    total_cases_planned = config.get("total_cases", 0)
    completed_cases = len(rows)
    passes = sum(1 for r in rows if r.get("preserved", True))
    failures = sum(1 for r in rows if not r.get("preserved", True))
    errors = sum(1 for r in rows if r.get("error"))
    elapsed = rows[-1]["cumulative_seconds"] if rows else 0.0
    # Determine strict preservation status and first failure
    strict_passed = all(r.get("preserved", True) and not r.get("error") for r in rows)
    first_failure = None
    for r in rows:
        if not r.get("preserved", True) or r.get("error"):
            first_failure = (r.get("cycle_n"), r.get("suspension_depth"))
            break
    # Build trajectory summary
    traj_summaries = build_trajectory_summary(rows)
    # Build failure rows
    failure_rows = []
    for r in rows:
        relabel_mismatch = r.get("relabel_checked") and not r.get("relabel_passed")
        if (not r.get("preserved", True)) or relabel_mismatch or r.get("error"):
            reason = []
            if not r.get("preserved", True):
                reason.append("status_mismatch")
            if relabel_mismatch:
                reason.append("relabel_mismatch")
            if r.get("error"):
                reason.append("error")
            failure_rows.append({
                "cycle_n": r.get("cycle_n"),
                "suspension_depth": r.get("suspension_depth"),
                "degree": r.get("degree"),
                "seed_trivial": r.get("seed_trivial"),
                "lifted_trivial": r.get("lifted_trivial"),
                "expected_by_parity": r.get("expected_by_parity"),
                "rank_A": r.get("rank_A"),
                "rank_augmented": r.get("rank_augmented"),
                "rank_gap": r.get("rank_gap"),
                "matrix_shape": f"{r.get('matrix_rows')}×{r.get('matrix_cols')}",
                "cochain_support_size": r.get("cochain_support_size"),
                "num_simplices_total": r.get("num_simplices_total"),
                "failure_reason": ",".join(reason),
            })
    # Build runtime profile summary per cycle_n
    profile_summary = []
    grouped: dict[int, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r.get("cycle_n"), []).append(r)
    for n_val, group in grouped.items():
        max_depth_completed = max(r.get("suspension_depth", 0) for r in group)
        max_simplices = max(r.get("num_simplices_total", 0) for r in group)
        matrix_rows_max = max(r.get("matrix_rows", 0) for r in group)
        matrix_cols_max = max(r.get("matrix_cols", 0) for r in group)
        total_seconds = max(r.get("cumulative_seconds", 0.0) for r in group)
        profile_summary.append({
            "cycle_n": n_val,
            "max_depth_completed": max_depth_completed,
            "max_simplices": max_simplices,
            "max_matrix_shape": f"{matrix_rows_max}×{matrix_cols_max}",
            "total_seconds": total_seconds,
        })
    # Format run config lines
    n_range = f"{config.get('n_min')}..{config.get('n_max')}"
    max_depth_cfg = config.get("max_depth")
    relabel_enabled = config.get("do_random_relabel", False)
    relabel_trials_cfg = config.get("relabel_trials", 0)
    run_mode_cfg = config.get("run_mode", "strict")
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    md_lines: list[str] = []
    md_lines.append("# Suspension Defect Experiment Receipt")
    md_lines.append("")
    md_lines.append("## Run Config")
    md_lines.append("")
    md_lines.append(f"- cycle_n range: {n_range}")
    md_lines.append(f"- max suspension depth: {max_depth_cfg}")
    md_lines.append(f"- total cases planned: {total_cases_planned}")
    md_lines.append(f"- relabel enabled: {relabel_enabled}")
    md_lines.append(f"- relabel trials per case: {relabel_trials_cfg}")
    md_lines.append(f"- run mode: {run_mode_cfg}")
    md_lines.append(f"- generated_at: {generated_at}")
    md_lines.append("")
    md_lines.append("## Completion")
    md_lines.append("")
    md_lines.append(f"- completed cases: {completed_cases} / {total_cases_planned}")
    md_lines.append(f"- passes: {passes}")
    md_lines.append(f"- failures: {failures}")
    md_lines.append(f"- errors: {errors}")
    md_lines.append(f"- elapsed seconds: {elapsed:.1f}")
    md_lines.append("")
    md_lines.append("## Strict Summary")
    md_lines.append("")
    md_lines.append(f"- strict_preservation_passed_so_far: {strict_passed}")
    if first_failure:
        md_lines.append(f"- first_failure: n={first_failure[0]}, depth={first_failure[1]}")
    else:
        md_lines.append(f"- first_failure: none")
    md_lines.append("")
    md_lines.append("## Trajectory Summary")
    md_lines.append("")
    if traj_summaries:
        md_lines.append("| n | seed | depths_completed | trajectory | flips | flip_depths | terminal |")
        md_lines.append("|---|------|------------------|------------|-------|-------------|----------|")
        for ts in traj_summaries:
            flip_depths_str = json.dumps(ts["flip_depths"]) if ts["flip_depths"] else "[]"
            md_lines.append(
                f"| {ts['cycle_n']} | {ts['seed_status_symbol']} | {ts['depths_completed']} | "
                f"{ts['trajectory_string']} | {ts['flip_count']} | {flip_depths_str} | {ts['terminal_status']} |"
            )
    else:
        md_lines.append("None.")
    md_lines.append("")
    md_lines.append("## Failure Rows")
    md_lines.append("")
    if failure_rows:
        md_lines.append(
            "| n | depth | degree | seed | lifted | expected | rank_A | rank_aug | rank_gap | matrix_shape | support_size | simplices | reason |"
        )
        md_lines.append(
            "|---|------|--------|------|--------|----------|-------|---------|----------|--------------|-------------|-----------|--------|"
        )
        for fr in failure_rows:
            md_lines.append(
                f"| {fr['cycle_n']} | {fr['suspension_depth']} | {fr['degree']} | "
                f"{fr['seed_trivial']} | {fr['lifted_trivial']} | {fr['expected_by_parity']} | "
                f"{fr['rank_A']} | {fr['rank_augmented']} | {fr['rank_gap']} | {fr['matrix_shape']} | "
                f"{fr['cochain_support_size']} | {fr['num_simplices_total']} | {fr['failure_reason']} |"
            )
    else:
        md_lines.append("None.")
    md_lines.append("")
    md_lines.append("## Runtime Profile")
    md_lines.append("")
    if profile_summary:
        md_lines.append("| n | max_depth_completed | max_simplices | max_matrix_shape | total_seconds |")
        md_lines.append("|---|---------------------|---------------|------------------|---------------|")
        for ps in sorted(profile_summary, key=lambda x: x.get("cycle_n")):
            md_lines.append(
                f"| {ps['cycle_n']} | {ps['max_depth_completed']} | {ps['max_simplices']} | "
                f"{ps['max_matrix_shape']} | {ps['total_seconds']:.4f} |"
            )
    else:
        md_lines.append("None.")
    md_lines.append("")
    md_lines.append("## Last Completed Row")
    md_lines.append("")
    if rows:
        last = rows[-1]
        last_summary = {
            "case_index": last.get("case_index"),
            "cycle_n": last.get("cycle_n"),
            "suspension_depth": last.get("suspension_depth"),
            "degree": last.get("degree"),
            "lifted_trivial": last.get("lifted_trivial"),
            "expected_by_parity": last.get("expected_by_parity"),
            "preserved": last.get("preserved"),
        }
        md_lines.append("```json")
        md_lines.append(json.dumps(last_summary, indent=2))
        md_lines.append("```")
    else:
        md_lines.append("None.")
    md_lines.append("")
    md_lines.append("## Row Schema")
    md_lines.append("")
    if rows:
        keys = list(rows[0].keys())
        md_lines.append("<ul>")
        for k in keys:
            md_lines.append(f"  <li>{k}</li>")
        md_lines.append("</ul>")
    else:
        md_lines.append("No rows to infer schema from.")
    md_lines.append("")
    return "\n".join(md_lines)


def write_exports(rows: list[dict], config: dict, export_dir: str = "/home/oai/share/exports") -> None:
    """Write export files for rows and receipts to a directory.

    This function writes four files into ``export_dir``:

      - ``rows.csv``: CSV representation of the rows
      - ``rows.jsonl``: JSON Lines representation of the rows (one JSON object per line)
      - ``trajectories.txt``: compact trajectory strings grouped by cycle_n
      - ``receipt.md``: human‑readable receipt rendered via ``render_markdown_receipt``

    Files are overwritten on each call. The export directory is created if
    it does not exist.

    Args:
        rows: list of result dictionaries
        config: run configuration dictionary
        export_dir: directory path for exports (default: ``/home/oai/share/exports``)
    """
    # Only attempt checkpointing if enabled
    if not ENABLE_DISK_CHECKPOINT:
        return
    try:
        os.makedirs(export_dir, exist_ok=True)
        # Write rows.csv and rows.jsonl
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(export_dir, "rows.csv"), index=False)
        with open(os.path.join(export_dir, "rows.jsonl"), "w", encoding="utf-8") as f_jsonl:
            for r in rows:
                f_jsonl.write(json.dumps(r) + "\n")
        # Write trajectories.txt
        traj_summaries = build_trajectory_summary(rows)
        with open(os.path.join(export_dir, "trajectories.txt"), "w", encoding="utf-8") as f_traj:
            for ts in traj_summaries:
                f_traj.write(f"{ts['cycle_n']}: {ts['trajectory_string']}\n")
        # Write receipt.md
        receipt_md = render_markdown_receipt(rows, config)
        with open(os.path.join(export_dir, "receipt.md"), "w", encoding="utf-8") as f_md:
            f_md.write(receipt_md)
    except PermissionError:
        # Silently ignore permission errors; checkpointing may not be
        # possible in restricted environments
        return


def main_app() -> None:
    """Render the Streamlit interface and handle user interactions.

    This simplified implementation provides only a single live run view.
    All computation results are accumulated into ``st.session_state.rows``,
    which serves as the canonical store for the full run. Exports are
    generated from this full store, regardless of any filtering used
    for display. File system writes are avoided by default; export
    downloads are generated in memory.
    """
    st.title("Suspension Defect Experiment (F₂)")
    st.markdown(
        "This app tests whether a 1D (F₂) parity defect on a cycle graph "
        "preserves its cohomology zero/nonzero status under repeated "
        "simplicial suspension. It is a toy model for defect‑lineage "
        "preservation, not a general theorem about knots."
    )
    # Initialise session state variables
    if "rows" not in st.session_state:
        st.session_state.rows = []
    if "run_in_progress" not in st.session_state:
        st.session_state.run_in_progress = False
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    # Sidebar configuration controls
    st.sidebar.header("Experiment Controls")
    # Range of cycle lengths
    n_min, n_max = st.sidebar.slider(
        "Cycle length range (inclusive)",
        min_value=3,
        max_value=20,
        value=(3, 10),
        step=1,
    )
    # Maximum suspension depth
    max_depth = st.sidebar.slider(
        "Max suspension depth", min_value=0, max_value=8, value=4, step=1
    )
    # Anti‑cheat relabel option
    do_random_relabel = st.sidebar.checkbox(
        "Enable random relabel anti‑cheat", value=False
    )
    # Relabel trials if anti‑cheat enabled
    relabel_trials = 1
    if do_random_relabel:
        relabel_trials = st.sidebar.number_input(
            "Number of relabel trials", min_value=1, max_value=10, value=1, step=1
        )
    # Stop after first failure option
    stop_after_failure = st.sidebar.checkbox(
        "Stop after first failure", value=False
    )
    # Compute total cases for the proposed run
    num_cycles = n_max - n_min + 1
    num_depths = max_depth + 1
    total_cases = num_cycles * num_depths
    # Show run configuration summary
    st.subheader("Run Configuration")
    st.write(f"Cycle lengths: {n_min}–{n_max} ({num_cycles} values)")
    st.write(f"Suspension depths: 0–{max_depth} ({num_depths} depths)")
    st.write(f"Total cases planned: {total_cases}")
    st.write(f"Relabel enabled: {do_random_relabel}")
    if do_random_relabel:
        st.write(f"Relabel trials per case: {relabel_trials}")
    # Large run warning
    LARGE_THRESHOLD = 200
    confirm_run = True
    if total_cases > LARGE_THRESHOLD:
        st.warning(
            "The selected range results in a large number of cases and may "
            "take a long time to compute."
        )
        confirm_run = st.checkbox("I understand this run may be slow.")
    # Placeholders for dynamic UI elements
    progress_bar = st.progress(0.0)
    status_area = st.empty()
    metrics_area = st.empty()
    table_area = st.empty()
    summary_area = st.empty()
    export_area = st.empty()
    # Run button: disabled if run in progress or not confirmed for large runs
    run_disabled = st.session_state.run_in_progress or not confirm_run
    if st.sidebar.button("Run test", disabled=run_disabled):
        # Clear previous results
        st.session_state.rows = []
        st.session_state.run_in_progress = True
        st.session_state.start_time = time.perf_counter()
        # Store config for receipt
        st.session_state.config = {
            "n_min": n_min,
            "n_max": n_max,
            "max_depth": max_depth,
            "do_random_relabel": do_random_relabel,
            "relabel_trials": int(relabel_trials),
            "stop_after_failure": stop_after_failure,
            "total_cases": total_cases,
            "run_mode": "strict" if stop_after_failure else "live",
        }
        # Iterate over experiment cases
        for row, prog in iter_experiment(
            n_min=n_min,
            n_max=n_max,
            max_depth=max_depth,
            do_random_relabel=do_random_relabel,
            relabel_trials=int(relabel_trials),
            early_stop_on_failure=stop_after_failure,
        ):
            # Append the current case row to the session state
            st.session_state.rows.append(row)
            # Optionally checkpoint to disk if enabled
            if ENABLE_DISK_CHECKPOINT:
                write_exports(st.session_state.rows, st.session_state.config)
            # Live run updates
            progress_bar.progress(prog["completed"] / prog["total"])
            elapsed = time.perf_counter() - st.session_state.start_time
            status_area.text(
                f"Case {prog['completed']}/{prog['total']} | n={row['cycle_n']} | depth={row['suspension_depth']} "
                f"| degree={row['degree']} | elapsed={elapsed:.1f}s | pass={prog['passes']} | fail={prog['failures']}"
            )
            metrics_area.markdown(
                f"**Current complexity**  "
                f"Vertices: {row['num_vertices']}  "
                f"Simplices: {row['num_simplices_total']}  "
                f"C^(q-1) basis size: {row['basis_size_prev']}  "
                f"C^q basis size: {row['basis_size_current']}  "
                f"Matrix shape: {row['matrix_rows']}×{row['matrix_cols']}"
            )
            # Show partial table
            table_area.dataframe(pd.DataFrame(st.session_state.rows))
            # Early stop on first failure
            if stop_after_failure and not row.get("preserved", True):
                break
        # Run completed or stopped early
        st.session_state.run_in_progress = False
        # Optionally write final exports to disk
        if ENABLE_DISK_CHECKPOINT:
            write_exports(st.session_state.rows, st.session_state.config)
        # Final progress bar
        progress_bar.progress(1.0)
        # Show completion message
        if not stop_after_failure or all(r.get("preserved", True) for r in st.session_state.rows):
            status_area.success("Run completed.")
        else:
            status_area.warning("Run stopped due to first failure.")
    # After run or during partial, display results
    rows = st.session_state.rows
    if rows:
        # Show final table and summary when run not in progress
        if not st.session_state.run_in_progress:
            df_full = pd.DataFrame(rows)
            table_area.dataframe(df_full)
            total_planned = st.session_state.config.get("total_cases", len(rows))
            passes = sum(1 for r in rows if r.get("preserved", True))
            failures = sum(1 for r in rows if not r.get("preserved", True))
            summary_area.markdown(
                f"**Summary:** total cases {len(rows)}/{total_planned}, passes {passes}, failures {failures}"
            )
            # Full export section
            export_area.subheader("Full Run Export")
            all_rows = rows  # canonical store for full run
            export_area.caption(f"Export includes {len(all_rows)} rows from the full live run.")
            # CSV
            export_area.download_button(
                "Download FULL rows.csv",
                data=make_rows_csv(all_rows),
                file_name="full_run_rows.csv",
                mime="text/csv",
            )
            # JSONL
            export_area.download_button(
                "Download FULL rows.jsonl",
                data=make_rows_jsonl(all_rows),
                file_name="full_run_rows.jsonl",
                mime="application/jsonl",
            )
            # Trajectories TXT
            export_area.download_button(
                "Download FULL trajectories.txt",
                data=make_trajectories_txt(all_rows),
                file_name="full_run_trajectories.txt",
                mime="text/plain",
            )
            # Markdown receipt
            receipt_text = render_markdown_receipt(all_rows, st.session_state.config)
            export_area.download_button(
                "Download FULL receipt.md",
                data=receipt_text,
                file_name="full_run_receipt.md",
                mime="text/markdown",
            )
            export_area.text_area("Copy/Paste FULL receipt", receipt_text, height=500)


if __name__ == "__main__":
    main_app()