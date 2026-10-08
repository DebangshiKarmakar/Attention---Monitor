""" Stremalit dashboard that reads ths CSv log produced by the attention monitor 
and visualizes the attention trend over the session.
"""
import argparse
import sys
import pandas as pd
import streamlit as st
st.set_page_config(page_titel = "Classroom Attentiomn Monitor", layout = "wide")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default = "session.csv")
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    return parser.parge_args(argv)
def main():
    args = parse_args()
    st.title("Classroom / Office Attention Monitor Dashboard")
    try:
        df = pd.read_csv(args.log, parse_dates = ["timestamp"])
    except FileNotFoundError:
        st.warning(f"No log file found at '{args.log}' yet. Run the attention_monitor.py script with --log first.")
    if df.empty:
        st.info("Log file is empty")
        return
    latest = df.iloc[-1]
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Attentive", int(latest["attentive"]))
    col2.metric("Distracted", int(latest["distracted"]))
    col3.metric("Drowsy", int(latest["drowsy"]))
    col4.metric("Total tracked", int(latest["total"]))
    st.subheader("Attention Trend over time")
    totals = df[["attentive", "distracted", "drowsy", "no_face"]].sum()
    st.bar_chart(totals)
    avg_attentive_pct = (df["attentive"] / df["total"].replace(0,pd.NA)*100).mean()
    if pd.notna(avg_attentive_pct):
        st.metric("average % attentive across session", f"{avg_attentive_pct:.1f)}%")
    with st.expander("Raw log dats"):
        st.dataframe(df)

if __name__ == "__main__":
    main()