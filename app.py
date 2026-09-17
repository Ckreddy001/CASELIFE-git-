"""
app.py -- Module 11: Interactive Dashboard (Streamlit)

Run with:  streamlit run app/app.py   (from the project root)

Shows: case information, top probable locations & probabilities,
movement analysis, anomalies, probable route, feature explanations,
and an interactive map -- all for a FICTIONAL, synthetic case.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from search_priority import compute_search_priority
from explainability import add_explanations
from mapping import build_investigation_map

st.set_page_config(page_title="CASEFILE: Missing Person Investigation Support", layout="wide")

st.title("🕵️ CASEFILE: AI-Powered Missing Person Investigation & Location Prediction")
st.caption(
    "Academic simulation only. Uses a fictional, synthetically generated case. "
    "Not for real-world use. Predictions are probabilistic and must never be treated as proof."
)

DATA = BASE / "data"
PROC = DATA / "processed"
REPORTS = BASE / "reports"


def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if Path(path).exists() else None


case = load_csv(DATA / "synthetic" / "synthetic_case_file.csv")
traj = load_csv(PROC / "trajectory_anomalies.csv", parse_dates=["timestamp"])
freq = load_csv(PROC / "frequently_visited_locations.csv")
route = load_csv(PROC / "predicted_route.csv")
model_compare = load_csv(REPORTS / "model_comparison.csv")

if case is None or traj is None:
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "python src/generate_synthetic_data.py\n"
        "python src/preprocessing.py\n"
        "python src/feature_engineering.py\n"
        "python src/clustering.py\n"
        "python src/anomaly_detection.py\n"
        "python src/prediction.py\n"
        "python src/route_prediction.py\n"
        "python src/search_priority.py\n"
        "python src/explainability.py"
    )
    st.stop()

# ---------------- Case Information ----------------
st.header("📋 Case Information")
c1, c2, c3 = st.columns(3)
c1.metric("Case ID", case["case_id"].iloc[0])
c2.metric("Subject", case["subject_id"].iloc[0])
c3.metric("Last Known Time", case["last_known_timestamp"].iloc[0])
st.info(case["notes"].iloc[0])

last_known = (case["last_known_latitude"].iloc[0], case["last_known_longitude"].iloc[0])

# ---------------- Search Priority ----------------
st.header("🎯 Top Probable Locations (Search Priority)")

route_probs = {}
if route is not None and "cluster" in route.columns:
    n = len(route)
    for i, cl in enumerate(route["cluster"]):
        route_probs[cl] = max(0.0, 1 - i / max(n, 1))

if freq is not None:
    ranking = compute_search_priority(
        freq_locations=freq,
        model_predictions=pd.DataFrame(columns=["cluster", "probability"]),
        route_probs=route_probs,
        last_known_lat=last_known[0],
        last_known_lon=last_known[1],
    )
    ranking = add_explanations(ranking)
    st.dataframe(
        ranking[["search_rank", "cluster", "visit_count", "search_priority_score", "explanation"]],
        use_container_width=True,
    )
else:
    ranking = None
    st.warning("Run clustering.py to generate frequently visited locations.")

# ---------------- Movement Analysis ----------------
st.header("📊 Movement Pattern Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Pings per hour of day")
    st.bar_chart(traj.groupby("hour").size())
with col2:
    st.subheader("Pings per weekday")
    st.bar_chart(traj.groupby("weekday").size())

st.subheader("Speed distribution (km/h)")
st.line_chart(traj["speed_kmh"].reset_index(drop=True))

# ---------------- Anomalies ----------------
st.header("⚠️ Detected Anomalies")
st.caption("An anomaly flag means the movement statistically differs from the subject's "
           "normal pattern. It does NOT imply suspicious or criminal behavior.")
anomalies = traj[traj.get("anomaly_consensus", 0) == 1]
st.write(f"{len(anomalies)} anomalous points out of {len(traj)} total (consensus of ≥2 detectors).")
st.dataframe(anomalies[["timestamp", "latitude", "longitude", "speed_kmh",
                         "iforest_anomaly", "lof_anomaly", "ocsvm_anomaly"]],
             use_container_width=True)

# ---------------- Probable Route ----------------
st.header("🧭 Probable Route")
if route is not None:
    st.dataframe(route, use_container_width=True)
else:
    st.warning("Run route_prediction.py to generate the probable route.")

# ---------------- Model Comparison ----------------
if model_compare is not None:
    st.header("🤖 Location Prediction Model Comparison")
    st.dataframe(model_compare, use_container_width=True)

# ---------------- Interactive Map ----------------
st.header("🗺️ Interactive Investigation Map")
m = build_investigation_map(
    last_known=last_known,
    frequent_locations=freq,
    route_df=route,
    anomalies_df=traj,
    search_priority_df=ranking,
)
components.html(m._repr_html_(), height=550, scrolling=False)

st.markdown("---")
st.caption(
    "This dashboard is part of an individual Advanced Machine Learning academic project. "
    "All data is fictional/synthetic. Never use ML output as proof of a person's real location."
)

