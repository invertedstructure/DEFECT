"""
Streamlit app for the F₂ suspension‑defect experiment.

This application provides a simple user interface to run the finite (F₂)
simplicial suspension experiment without requiring any familiarity with
Python.  Users may select a range of cycle lengths, a maximum suspension
depth, and optionally enable a random relabel anti‑cheat check.  Upon
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

# -----------------------------------------------------------------------------
# Utility functions for summarising and exporting experiment results
# -----------------------------------------------------------------------------

def build_trajectory_summary(rows: list[dict]) -> list[dict]:
    """Compute a trajectory summary for each cycle_n based on a list of rows.

    Each row is assumed to contain at least the keys 'cycle_n',
    'suspension_depth' and 'status_symbol'.  The summary captures the
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


def render_markdown_receipt(rows: list[dict], config: dict) -> str:
    """Render a human‑readable Markdown receipt from result rows and config.

    The receipt summarises the run configuration, completion statistics,
    trajectory summary, failure rows, runtime profile, the last completed
    row, and the row schema.  It is deterministic and safe for copying
    into another document or chat.

    Args:
        rows: list of result dictionaries from ``iter_experiment``.
        config: dictionary containing run configuration parameters.  The keys
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

    Files are overwritten on each call.  The export directory is created if
    it does not exist.

    Args:
        rows: list of result dictionaries
        config: run configuration dictionary
        export_dir: directory path for exports (default: ``/home/oai/share/exports``)
    """
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


def main_app() -> None:
    """Render the Streamlit interface and handle user interactions.

    This implementation separates computation from presentation.  When the
    user initiates a run, all cases are computed once via ``iter_experiment``
    and the resulting rows are stored in ``st.session_state``.  Subsequent
    display modes (strict summary, trajectory summary, failure map, runtime
    profile, receipt export) derive their information solely from the stored
    rows.  Partial runs (e.g. stopping after the first failure) still
    populate the rows list and allow exports.  During long runs, partial
    results are checkpointed to disk under the ``exports`` directory.
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
    # Display mode selector
    display_mode = st.sidebar.selectbox(
        "Display mode",
        options=[
            "Live run",
            "Strict summary",
            "Trajectory summary",
            "Failure map",
            "Runtime profile",
            "Receipt export",
        ],
        index=0,
    )
    # Placeholders for dynamic UI elements
    progress_bar = st.progress(0.0)
    status_area = st.empty()
    metrics_area = st.empty()
    table_area = st.empty()
    summary_area = st.empty()
    download_area = st.empty()
    extra_area = st.empty()
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
            # record run mode (strict or live) for receipt
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
            # Append row
            st.session_state.rows.append(row)
            # Write partial exports to disk after each row
            write_exports(st.session_state.rows, st.session_state.config)
            # Live run updates
            if display_mode == "Live run":
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
                df_partial = pd.DataFrame(st.session_state.rows)
                table_area.dataframe(df_partial)
            # Early stop on first failure
            if stop_after_failure and not row.get("preserved", True):
                break
        # Run completed or stopped early
        st.session_state.run_in_progress = False
        # Final write to ensure exports reflect completion state
        write_exports(st.session_state.rows, st.session_state.config)
        # Signal completion
        if not stop_after_failure or all(r.get("preserved", True) for r in st.session_state.rows):
            status_area.success("Run completed.")
        else:
            status_area.warning("Run stopped due to first failure.")
    # After run or during partial, display according to mode
    rows = st.session_state.rows
    if rows:
        # Completed or partial data present
        if display_mode == "Live run":
            # If run has finished, display final table and summary
            if not st.session_state.run_in_progress:
                progress_bar.progress(1.0)
                df_final = pd.DataFrame(rows)
                table_area.dataframe(df_final)
                total = rows[-1].get("total_cases", len(rows))
                passes = sum(1 for r in rows if r.get("preserved", True))
                failures = sum(1 for r in rows if not r.get("preserved", True))
                summary_area.markdown(
                    f"**Summary:** total cases {len(rows)}/{total}, passes {passes}, failures {failures}"
                )
        elif display_mode == "Strict summary":
            total_cases_run = rows[-1].get("total_cases", len(rows))
            passes = sum(1 for r in rows if r.get("preserved", True))
            failures = sum(1 for r in rows if not r.get("preserved", True))
            errors = sum(1 for r in rows if r.get("error"))
            first_failure_row = next((r for r in rows if not r.get("preserved", True) or r.get("error")), None)
            strict_passed_so_far = (failures == 0 and errors == 0)
            summary_area.subheader("Strict Summary")
            summary_area.write(f"Completed cases: {len(rows)} / {total_cases_run}")
            summary_area.write(f"Passes: {passes}")
            summary_area.write(f"Failures: {failures}")
            summary_area.write(f"Errors: {errors}")
            summary_area.write(f"Strict preservation passed so far: {strict_passed_so_far}")
            if first_failure_row:
                summary_area.write(
                    f"First failure at n={first_failure_row['cycle_n']}, depth={first_failure_row['suspension_depth']}"
                )
            df = pd.DataFrame(rows)
            table_area.dataframe(df)
        elif display_mode == "Trajectory summary":
            traj_summaries = build_trajectory_summary(rows)
            summary_area.subheader("Trajectory Summary")
            if traj_summaries:
                df_traj = pd.DataFrame(traj_summaries)
                table_area.dataframe(df_traj)
                strict_count = sum(1 for ts in traj_summaries if ts["strict_preserved_all_completed"])
                non_strict_count = len(traj_summaries) - strict_count
                extra_area.write(f"Strictly preserved sequences: {strict_count}, other sequences: {non_strict_count}")
            else:
                summary_area.write("No data to summarise.")
        elif display_mode == "Failure map":
            failure_records = []
            for r in rows:
                relabel_mismatch = r.get("relabel_checked") and not r.get("relabel_passed")
                if (not r.get("preserved", True)) or relabel_mismatch or r.get("error"):
                    reason_parts = []
                    if not r.get("preserved", True):
                        reason_parts.append("status_mismatch")
                    if relabel_mismatch:
                        reason_parts.append("relabel_mismatch")
                    if r.get("error"):
                        reason_parts.append("error")
                    failure_records.append({
                        "cycle_n": r.get("cycle_n"),
                        "suspension_depth": r.get("suspension_depth"),
                        "degree": r.get("degree"),
                        "lifted_trivial": r.get("lifted_trivial"),
                        "expected_by_parity": r.get("expected_by_parity"),
                        "rank_gap": r.get("rank_gap"),
                        "matrix_shape": f"{r.get('matrix_rows')}×{r.get('matrix_cols')}",
                        "support_size": r.get("cochain_support_size"),
                        "num_simplices_total": r.get("num_simplices_total"),
                        "reason": ",".join(reason_parts),
                    })
            summary_area.subheader("Failure Map")
            if failure_records:
                df_fail = pd.DataFrame(failure_records)
                table_area.dataframe(df_fail)
            else:
                summary_area.write("No failures detected.")
        elif display_mode == "Runtime profile":
            # Build runtime profile per n
            profile_records = []
            grouped_runtime: dict[int, list[dict]] = {}
            for r in rows:
                grouped_runtime.setdefault(r.get("cycle_n"), []).append(r)
            for n_val, group in grouped_runtime.items():
                max_depth_completed = max(r.get("suspension_depth", 0) for r in group)
                max_simplices = max(r.get("num_simplices_total", 0) for r in group)
                matrix_rows_max = max(r.get("matrix_rows", 0) for r in group)
                matrix_cols_max = max(r.get("matrix_cols", 0) for r in group)
                total_seconds = max(r.get("cumulative_seconds", 0.0) for r in group)
                profile_records.append({
                    "cycle_n": n_val,
                    "max_depth_completed": max_depth_completed,
                    "max_simplices": max_simplices,
                    "max_matrix_shape": f"{matrix_rows_max}×{matrix_cols_max}",
                    "total_seconds": total_seconds,
                })
            summary_area.subheader("Runtime Profile")
            df_profile = pd.DataFrame(profile_records)
            table_area.dataframe(df_profile)
            # Provide a simple chart of case_seconds over run order
            if rows:
                extra_area.line_chart(pd.DataFrame({"case_seconds": [r.get("case_seconds") for r in rows]}))
        elif display_mode == "Receipt export":
            # Render the Markdown receipt and show it
            md = render_markdown_receipt(rows, st.session_state.config)
            summary_area.subheader("Receipt")
            summary_area.markdown(md, unsafe_allow_html=True)
            # Provide download buttons for exports
            df_rows = pd.DataFrame(rows)
            download_area.download_button(
                "Download rows CSV",
                data=df_rows.to_csv(index=False).encode("utf-8"),
                file_name="rows.csv",
                mime="text/csv",
            )
            jsonl_text = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
            download_area.download_button(
                "Download rows JSONL",
                data=jsonl_text,
                file_name="rows.jsonl",
                mime="application/jsonl",
            )
            traj_lines = []
            for ts in build_trajectory_summary(rows):
                traj_lines.append(f"{ts['cycle_n']}: {ts['trajectory_string']}")
            traj_bytes = "\n".join(traj_lines).encode("utf-8")
            download_area.download_button(
                "Download trajectories TXT",
                data=traj_bytes,
                file_name="trajectories.txt",
                mime="text/plain",
            )
            md_bytes = md.encode("utf-8")
            download_area.download_button(
                "Download receipt MD",
                data=md_bytes,
                file_name="receipt.md",
                mime="text/markdown",
            )
    else:
        summary_area.info("Adjust the parameters in the sidebar and click 'Run test' to begin.")


def main():
    """Render the Streamlit interface and handle user interactions with progress."""
    st.title("Suspension Defect Experiment (F₂)")
    st.markdown(
        "This app tests whether a 1D (F₂) parity defect on a cycle graph "
        "preserves its cohomology zero/nonzero status under repeated "
        "simplicial suspension. It is a toy model for defect‑lineage "
        "preservation, not a general theorem about knots."
    )

    # Sidebar controls
    st.sidebar.header("Experiment Controls")
    # Run mode selection
    run_mode = st.sidebar.selectbox(
        "Run mode",
        options=[
            "Strict preservation",
            "Trajectory / lineage",
            "Failure map",
            "Runtime profile",
            "Demo / sanity",
        ],
        index=0,
    )
    # Cycle range selector with defaults depending on mode
    default_cycle_range = (3, 10) if run_mode != "Demo / sanity" else (3, 10)
    n_min, n_max = st.sidebar.slider(
        "Cycle length range (inclusive)",
        min_value=3,
        max_value=20,
        value=default_cycle_range,
        step=1,
    )
    # Max suspension depth
    default_max_depth = 4 if run_mode != "Demo / sanity" else 4
    max_depth = st.sidebar.slider(
        "Max suspension depth", min_value=0, max_value=8, value=default_max_depth, step=1
    )
    # Anti‑cheat toggle (only meaningful for strict and trajectory modes but allow always)
    do_random_relabel = st.sidebar.checkbox(
        "Enable random relabel anti‑cheat", value=False
    )
    # Number of relabel trials if anti‑cheat
    relabel_trials = 1
    if do_random_relabel:
        relabel_trials = st.sidebar.number_input(
            "Number of relabel trials", min_value=1, max_value=10, value=1, step=1
        )
    # Stop after first failure (only applies to strict mode)
    stop_after_failure = False
    if run_mode == "Strict preservation":
        stop_after_failure = st.sidebar.checkbox(
            "Stop after first failure", value=False
        )

    # Pre‑run summary
    st.subheader("Run Configuration")
    num_cycles = n_max - n_min + 1
    num_depths = max_depth + 1
    total_cases = num_cycles * num_depths
    st.write(f"Cycle lengths: {n_min}–{n_max} ({num_cycles} values)")
    st.write(f"Suspension depths: 0–{max_depth} ({num_depths} depths)")
    st.write(f"Total cases: {total_cases}")
    st.write(f"Anti‑cheat relabel enabled: {do_random_relabel}")
    if do_random_relabel:
        st.write(f"Relabel trials per case: {relabel_trials}")
    # Large run warning
    LARGE_THRESHOLD = 200
    confirm = True
    if total_cases > LARGE_THRESHOLD:
        st.warning(
            "The selected range results in a large number of cases and may "
            "take a long time to compute."
        )
        confirm = st.checkbox("I understand this run may be slow.")

    # Placeholders for progress and dynamic outputs
    progress_bar = st.progress(0.0)
    status_area = st.empty()
    latest_area = st.empty()
    table_area = st.empty()
    metrics_area = st.empty()
    failure_warning = st.empty()
    failure_details_area = st.empty()
    download_area = st.empty()
    extra_table_area = st.empty()
    trajectory_summary_area = st.empty()

    # Define a helper to classify trajectory sequences
    def classify_trajectory(seed_status: bool, statuses: list) -> str:
        """Classify a trajectory sequence into one of several labels."""
        # If any error status present
        if any(s is None for s in statuses):
            return "ERROR"
        # Convert booleans to string labels for clarity (not used here)
        # Always trivial or always nontrivial
        if all(statuses):
            return "ALWAYS_TRIVIAL"
        if not any(statuses):
            return "ALWAYS_NONTRIVIAL"
        # Strictly preserved (all equal to seed)
        if all(s == seed_status for s in statuses):
            return "STRICTLY_PRESERVED"
        # Identify mismatch segments (contiguous stretches of non-seed status)
        mismatch_segments = []
        i = 0
        n = len(statuses)
        while i < n:
            if statuses[i] != seed_status:
                start = i
                while i < n and statuses[i] != seed_status:
                    i += 1
                end = i
                mismatch_segments.append((start, end))
            else:
                i += 1
        # Count returns to seed status
        recurrence_count = 0
        in_mismatch = False
        for s in statuses:
            if not in_mismatch and s != seed_status:
                in_mismatch = True
            elif in_mismatch and s == seed_status:
                recurrence_count += 1
                in_mismatch = False
        # Mismatch-only-once: exactly one mismatch segment of length 1 and returns once
        if len(mismatch_segments) == 1 and (mismatch_segments[0][1] - mismatch_segments[0][0]) == 1 and recurrence_count == 1:
            return "MISMATCH_ONLY_ONCE"
        # Dies forever: mismatches occur and no returns
        if recurrence_count == 0:
            return "DIES_FOREVER"
        # Dies then returns: exactly one return
        if recurrence_count == 1:
            return "DIES_THEN_RETURNS"
        # Oscillates: multiple returns
        if recurrence_count > 1:
            return "OSCILLATES"
        # Fallback
        return "ERROR"

    # Run button - disabled if not confirmed for large runs
    run_disabled = not confirm
    if st.sidebar.button("Run test", disabled=run_disabled):
        # Reset output areas
        failure_warning.empty()
        failure_details_area.empty()
        download_area.empty()
        extra_table_area.empty()
        trajectory_summary_area.empty()
        # Initialize containers
        results = []
        failures = []
        passes_count = 0
        fails_count = 0
        start_time = time.perf_counter()
        # Determine early stop flag based on mode
        early_stop = stop_after_failure if run_mode == "Strict preservation" else False
        # Iterate over experiment cases
        for row, prog in iter_experiment(
            n_min=n_min,
            n_max=n_max,
            max_depth=max_depth,
            do_random_relabel=do_random_relabel,
            relabel_trials=int(relabel_trials),
            early_stop_on_failure=early_stop,
        ):
            # Append row to results
            results.append(row)
            # Update counters
            passes_count = prog["passes"]
            fails_count = prog["failures"]
            completed = prog["completed"]
            total = prog["total"]
            # Update progress bar
            progress_bar.progress(completed / total)
            # Update status text
            elapsed = time.perf_counter() - start_time
            status_area.text(
                f"Case {completed}/{total} | n={row['cycle_n']} | depth={row['suspension_depth']} "
                f"| degree={row['degree']} | elapsed={elapsed:.1f}s | pass={passes_count} | fail={fails_count}"
            )
            # Update latest result
            latest_area.markdown(
                f"Last completed case: n={row['cycle_n']}, depth={row['suspension_depth']}, "
                f"lifted_trivial={row['lifted_trivial']}, expected={row['expected_by_parity']}, preserved={row['preserved']}"
            )
            # Update complexity metrics display
            metrics_area.markdown(
                f"**Current complexity**  \
                Vertices: {row['num_vertices']}  \
                Simplices: {row['num_simplices_total']}  \
                Basis size C^{{q-1}}: {row['basis_size_prev']}  \
                Basis size C^q: {row['basis_size_current']}  \
                Matrix shape: {row['matrix_rows']} × {row['matrix_cols']}"
            )
            # Update partial table (refresh every case)
            df_partial = pd.DataFrame(results)
            table_area.dataframe(df_partial)
            # Handle failure detection
            if not row['preserved']:
                failures.append(row.get('failure_details', {}))
                failure_warning.warning(
                    "Failure detected: the cohomology status did not match the expected parity."
                )
                # Show first failure details
                if failures:
                    with failure_details_area.expander("Failure Diagnostics", expanded=True):
                        st.json(failures[0])
                # Stop if user requested early stop in strict mode
                if early_stop:
                    break
        # After iteration completes
        df_final = pd.DataFrame(results)
        if run_mode == "Strict preservation" or run_mode == "Demo / sanity":
            # Compute summary
            total_cases_done = len(results)
            summary = {
                "total_cases": total_cases_done,
                "passes": passes_count,
                "failures": fails_count,
                "even_cases": len([r for r in results if r['cycle_n'] % 2 == 0]),
                "even_failures": len([r for r in results if r['cycle_n'] % 2 == 0 and not r['preserved']]),
                "odd_cases": len([r for r in results if r['cycle_n'] % 2 == 1]),
                "odd_failures": len([r for r in results if r['cycle_n'] % 2 == 1 and not r['preserved']]),
            }
            # Display final summary
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total cases", summary["total_cases"])
            col2.metric("Passes", summary["passes"])
            col3.metric("Failures", summary["failures"])
            st.write(
                f"Even cycles: {summary['even_cases']} cases, {summary['even_failures']} failures"
            )
            st.write(
                f"Odd cycles: {summary['odd_cases']} cases, {summary['odd_failures']} failures"
            )
            # Offer CSV download for row-level data
            csv_data = df_final.to_csv(index=False).encode("utf-8")
            download_area.download_button(
                "Download row-level results CSV",
                data=csv_data,
                file_name="suspension_defect_results.csv",
                mime="text/csv",
            )
        elif run_mode == "Trajectory / lineage":
            # Build trajectory sequences by n
            sequences = {}
            for r in results:
                sequences.setdefault(r['cycle_n'], []).append('T' if r['lifted_trivial'] else 'N')
            trajectory_rows = []
            for n_val, seq_chars in sequences.items():
                seed_status_char = seq_chars[0]
                seed_status_bool = (seed_status_char == 'T')
                # Convert seq chars to boolean list
                bool_seq = [c == 'T' for c in seq_chars]
                # compute first failure and first return
                first_failure = next((i for i, s in enumerate(bool_seq) if s != seed_status_bool), None)
                first_return = None
                recurrence = 0
                in_mismatch = False
                for i, s in enumerate(bool_seq):
                    if i == 0:
                        continue
                    if not in_mismatch and s != seed_status_bool:
                        in_mismatch = True
                    elif in_mismatch and s == seed_status_bool:
                        recurrence += 1
                        if first_return is None:
                            first_return = i
                        in_mismatch = False
                # longest trivial gap: longest consecutive run of True statuses
                longest_trivial_gap = 0
                current_gap = 0
                for s in bool_seq:
                    if s:
                        current_gap += 1
                        if current_gap > longest_trivial_gap:
                            longest_trivial_gap = current_gap
                    else:
                        current_gap = 0
                terminal_status_char = 'T' if bool_seq[-1] else 'N'
                traj_label = classify_trajectory(seed_status_bool, bool_seq)
                trajectory_rows.append({
                    'cycle_n': n_val,
                    'seed_status': seed_status_char,
                    'trajectory_by_depth': ''.join(seq_chars),
                    'first_failure_depth': first_failure,
                    'first_return_depth': first_return,
                    'recurrence_count': recurrence,
                    'longest_trivial_gap': longest_trivial_gap,
                    'terminal_status': terminal_status_char,
                    'trajectory_label': traj_label,
                })
            df_traj = pd.DataFrame(trajectory_rows)
            # Show row-level table
            st.subheader("Row-level Results")
            table_area.dataframe(df_final)
            # Show trajectory summary table
            st.subheader("Trajectory Summary by n")
            extra_table_area.dataframe(df_traj)
            # Summary of labels
            label_counts = df_traj['trajectory_label'].value_counts().to_dict()
            summary_lines = [f"{label}: {count}" for label, count in label_counts.items()]
            trajectory_summary_area.markdown("**Trajectory classification counts**<br>" + "<br>".join(summary_lines), unsafe_allow_html=True)
            # CSV downloads for row-level and trajectory-level
            csv_rows = df_final.to_csv(index=False).encode("utf-8")
            csv_traj = df_traj.to_csv(index=False).encode("utf-8")
            download_area.download_button(
                "Download row-level results CSV",
                data=csv_rows,
                file_name="suspension_defect_row_results.csv",
                mime="text/csv",
            )
            download_area.download_button(
                "Download trajectory-level results CSV",
                data=csv_traj,
                file_name="suspension_defect_trajectory_results.csv",
                mime="text/csv",
            )
        elif run_mode == "Failure map":
            # Extract failure rows
            failure_records = []
            for r in results:
                if not r['preserved'] or (r['relabel_checked'] and not r['relabel_passed']):
                    reason = 'Status mismatch'
                    if r['relabel_checked'] and not r['relabel_passed']:
                        reason = 'Relabel mismatch'
                    details = r.get('failure_details', {})
                    failure_records.append({
                        'cycle_n': r['cycle_n'],
                        'suspension_depth': r['suspension_depth'],
                        'degree': r['degree'],
                        'seed_trivial': r['seed_trivial'],
                        'lifted_trivial': r['lifted_trivial'],
                        'expected_by_parity': r['expected_by_parity'],
                        'rank_A': r['rank_A'],
                        'rank_augmented': r['rank_augmented'],
                        'matrix_shape': f"{r['matrix_rows']}×{r['matrix_cols']}",
                        'cochain_support_size': r['cochain_support_size'],
                        'num_simplices_total': r['num_simplices_total'],
                        'failure_reason': reason,
                    })
            df_fail = pd.DataFrame(failure_records)
            if df_fail.empty:
                st.success("No failures detected in the selected range.")
            else:
                st.subheader("Failure Map")
                extra_table_area.dataframe(df_fail)
                csv_fail = df_fail.to_csv(index=False).encode("utf-8")
                download_area.download_button(
                    "Download failure map CSV",
                    data=csv_fail,
                    file_name="suspension_defect_failures.csv",
                    mime="text/csv",
                )
        elif run_mode == "Runtime profile":
            # Focus on timings and complexity
            profile_records = []
            for r in results:
                profile_records.append({
                    'cycle_n': r['cycle_n'],
                    'suspension_depth': r['suspension_depth'],
                    'degree': r['degree'],
                    'num_vertices': r['num_vertices'],
                    'num_simplices_total': r['num_simplices_total'],
                    'basis_size_prev': r['basis_size_prev'],
                    'basis_size_current': r['basis_size_current'],
                    'matrix_shape': f"{r['matrix_rows']}×{r['matrix_cols']}",
                    'case_seconds': r['case_seconds'],
                    'cumulative_seconds': r['cumulative_seconds'],
                })
            df_profile = pd.DataFrame(profile_records)
            st.subheader("Runtime Profile")
            extra_table_area.dataframe(df_profile)
            # Plot simple charts by depth (x-axis: combined n depth maybe index)
            # Chart 1: case_seconds by index
            st.line_chart(df_profile['case_seconds'])
            # Chart 2: num_simplices_total by index
            st.line_chart(df_profile['num_simplices_total'])
            # Chart 3: matrix rows vs case index
            st.line_chart(df_profile['num_vertices'])
            csv_profile = df_profile.to_csv(index=False).encode("utf-8")
            download_area.download_button(
                "Download runtime profile CSV",
                data=csv_profile,
                file_name="suspension_defect_runtime_profile.csv",
                mime="text/csv",
            )
        else:
            # Should not reach here
            pass
    else:
        st.info(
            "Adjust the parameters in the sidebar and click 'Run test' to begin."
        )


if __name__ == "__main__":
    # Invoke the new main application function rather than the legacy main.
    main_app()