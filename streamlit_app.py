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
from defect_core import run_experiment


def main():
    """Render the Streamlit interface and handle user interactions."""
    st.title("Suspension Defect Experiment (F₂)")
    st.markdown(
        "This app tests whether a 1D (F₂) parity defect on a cycle graph "
        "preserves its cohomology zero/nonzero status under repeated "
        "simplicial suspension. It is a toy model for defect‑lineage "
        "preservation, not a general theorem about knots."
    )

    st.sidebar.header("Experiment Controls")
    # Cycle length range selection
    n_min, n_max = st.sidebar.slider(
        "Cycle length range (inclusive)",
        min_value=3,
        max_value=20,
        value=(3, 10),
        step=1,
    )
    # Suspension depth selection
    max_depth = st.sidebar.slider(
        "Max suspension depth", min_value=0, max_value=8, value=4, step=1
    )
    # Random relabel anti‑cheat
    do_random_relabel = st.sidebar.checkbox(
        "Perform random relabel anti‑cheat check", value=True
    )

    # Run button
    if st.sidebar.button("Run test"):
        # Execute experiment
        with st.spinner("Running experiment..."):
            results, summary, failures = run_experiment(
                n_min=n_min,
                n_max=n_max,
                max_depth=max_depth,
                do_random_relabel=do_random_relabel,
            )
        # Display summary metrics
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
        # Show result table
        df = pd.DataFrame(results)
        st.subheader("Detailed Results")
        st.dataframe(df)
        # Show warnings if there are failures
        if summary["failures"] > 0:
            st.warning(
                "Failures detected: some cases did not preserve the expected "
                "trivial/nontrivial status. See diagnostics below."
            )
        # Diagnostics expander
        if failures:
            with st.expander("Failure Diagnostics"):
                for idx, fail in enumerate(failures, start=1):
                    st.markdown(f"### Failure {idx}")
                    st.json(fail)
    else:
        st.info("Adjust the parameters in the sidebar and click 'Run test' to begin.")


if __name__ == "__main__":
    main()