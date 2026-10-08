"""Streamlit dashboard for attention-monitor session logs."""

import argparse
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classroom Attention Monitor", layout="wide")

ATTENTION_COLUMNS = ["attentive", "distracted", "drowsy", "no_face"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="session.csv")
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    st.title("Classroom / Office Attention Monitor Dashboard")
    try:
        df = pd.read_csv(args.log, parse_dates=["timestamp"])
    except FileNotFoundError:
        st.warning(
            f"No log file found at '{args.log}' yet. "
            "Run attention_monitor.py with --log first."
        )
        return
    except pd.errors.EmptyDataError:
        st.info("Log file is empty.")
        return

    if df.empty:
        st.info("Log file has no session data yet.")
        return

    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attentive", int(latest["attentive"]))
    col2.metric("Distracted", int(latest["distracted"]))
    col3.metric("Drowsy", int(latest["drowsy"]))
    col4.metric("Total tracked", int(latest["total"]))

    st.subheader("Attention Trend Over Time")
    trend = df.set_index("timestamp")[ATTENTION_COLUMNS]
    st.line_chart(trend)

    nonzero_totals = df["total"].replace(0, float("nan"))
    avg_attentive_pct = (df["attentive"] / nonzero_totals * 100).mean()
    if pd.notna(avg_attentive_pct):
        st.metric("Average attentive", f"{avg_attentive_pct:.1f}%")

    with st.expander("Raw log data"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
