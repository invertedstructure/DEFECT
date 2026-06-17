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
from defect_core import run_experiment, iter_experiment

import time


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
    main()