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
    # Cycle range selector
    n_min, n_max = st.sidebar.slider(
        "Cycle length range (inclusive)",
        min_value=3,
        max_value=20,
        value=(3, 10),
        step=1,
    )
    # Max suspension depth
    max_depth = st.sidebar.slider(
        "Max suspension depth", min_value=0, max_value=8, value=4, step=1
    )
    # Anti‑cheat toggle
    do_random_relabel = st.sidebar.checkbox(
        "Enable random relabel anti‑cheat", value=False
    )
    # Number of relabel trials if anti‑cheat
    relabel_trials = 1
    if do_random_relabel:
        relabel_trials = st.sidebar.number_input(
            "Number of relabel trials", min_value=1, max_value=10, value=1, step=1
        )
    # Stop after first failure
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

    # Run button - disabled if not confirmed for large runs
    run_disabled = not confirm
    if st.sidebar.button("Run test", disabled=run_disabled):
        # Reset output areas
        failure_warning.empty()
        failure_details_area.empty()
        download_area.empty()
        # Initialize containers
        results = []
        failures = []
        passes = 0
        fails = 0
        start_time = time.perf_counter()
        # Iterate over experiment cases
        for row, prog in iter_experiment(
            n_min=n_min,
            n_max=n_max,
            max_depth=max_depth,
            do_random_relabel=do_random_relabel,
            relabel_trials=int(relabel_trials),
            early_stop_on_failure=stop_after_failure,
        ):
            # Append row to results
            results.append(row)
            # Update counters
            passes = prog["passes"]
            fails = prog["failures"]
            completed = prog["completed"]
            # Update progress bar
            progress_bar.progress(completed / prog["total"])
            # Update status text
            elapsed = time.perf_counter() - start_time
            status_area.text(
                f"Case {completed}/{prog['total']} | n={row['cycle_n']} | depth={row['suspension_depth']} "
                f"| degree={row['degree']} | elapsed={elapsed:.1f}s | pass={passes} | fail={fails}"
            )
            # Update latest result
            latest_area.markdown(
                f"Last completed case: n={row['cycle_n']}, depth={row['suspension_depth']}, "
                f"lifted_trivial={row['lifted_trivial']}, expected={row['expected_by_parity']}, preserved={row['preserved']}"
            )
            # Update complexity metrics display
            metrics_area.markdown(
                f"**Current complexity**  \
                Vertices: {row['vertices']}  \
                Simplices: {row['simplices_total']}  \
                Basis size C^{{q-1}}: {row['basis_prev_size']}  \
                Basis size C^q: {row['basis_curr_size']}  \
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
                # Stop if user requested early stop
                if stop_after_failure:
                    break
        # After iteration completes
        total_cases_done = len(results)
        summary = {
            "total_cases": total_cases_done,
            "passes": passes,
            "failures": fails,
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
        # Offer CSV download
        df_final = pd.DataFrame(results)
        csv_data = df_final.to_csv(index=False).encode("utf-8")
        download_area.download_button(
            "Download results CSV",
            data=csv_data,
            file_name="suspension_defect_results.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Adjust the parameters in the sidebar and click 'Run test' to begin."
        )


if __name__ == "__main__":
    main()