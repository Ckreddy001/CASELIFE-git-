"""
CASEFILE: AI-Powered Missing Person Investigation and Probable Location Prediction System
Academic simulation using synthetic/publicly available non-PII data.

Run:
    pip install -r requirements.txt
    streamlit run casefile_app.py

The script can:
1. Generate a synthetic movement dataset
2. Preprocess and engineer movement features
3. Perform EDA
4. Perform K-Means clustering
5. Perform Isolation Forest anomaly detection
6. Train a next-location prediction model
7. Build a movement graph and predict candidate routes
8. Calculate search-priority scores
9. Provide model explanations using permutation importance
10. Display an interactive Streamlit dashboard with maps and charts

Important: This is an educational investigation-support simulation. It does not identify,
track, or predict the real location of any person.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.inspection import permutation_importance

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

try:
    import networkx as nx
    NETWORKX_OK = True
except Exception:
    NETWORKX_OK = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="CASEFILE AI Investigation System",
    page_icon="🔎",
    layout="wide"
)

RANDOM_STATE = 42
DATA_DIR = "data"
MODEL_DIR = "models"
MAP_DIR = "maps"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------

def generate_synthetic_dataset(n_people=30, days=30, points_per_day=20):
    """
    Generate fictional movement records.

    The dataset intentionally contains no real-person data.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Fictional city center. These coordinates are only used as synthetic
    # geographic coordinates and do not represent a real case.
    base_lat = 23.25
    base_lon = 77.41

    locations = {
        "Home": (base_lat + 0.010, base_lon + 0.010),
        "College": (base_lat + 0.035, base_lon + 0.020),
        "Office": (base_lat - 0.020, base_lon + 0.040),
        "Market": (base_lat + 0.005, base_lon - 0.030),
        "Station": (base_lat - 0.040, base_lon - 0.010),
        "Park": (base_lat + 0.045, base_lon - 0.025),
        "Hospital": (base_lat - 0.010, base_lon - 0.050),
        "Mall": (base_lat + 0.025, base_lon + 0.055),
        "Unknown_A": (base_lat + 0.065, base_lon + 0.075),
        "Unknown_B": (base_lat - 0.070, base_lon + 0.065),
        "Unknown_C": (base_lat + 0.080, base_lon - 0.070),
    }

    normal_places = [
        "Home", "College", "Office", "Market",
        "Station", "Park", "Hospital", "Mall"
    ]

    records = []

    start_date = pd.Timestamp("2026-01-01")

    for person_num in range(1, n_people + 1):
        person_id = f"P{person_num:03d}"

        for day in range(days):
            date = start_date + pd.Timedelta(days=day)

            for p in range(points_per_day):
                hour = int((p / points_per_day) * 24)

                # Mostly normal daily behavior
                if 7 <= hour <= 9:
                    place = rng.choice(["Home", "College", "Office"])
                elif 10 <= hour <= 16:
                    place = rng.choice(["College", "Office", "Market", "Hospital"])
                elif 17 <= hour <= 20:
                    place = rng.choice(["Market", "Mall", "Park", "Home"])
                else:
                    place = "Home"

                # Occasionally generate a synthetic unusual movement
                # so that anomaly detection has something meaningful to find.
                is_unusual = rng.random() < 0.04
                if is_unusual:
                    place = rng.choice(["Unknown_A", "Unknown_B", "Unknown_C"])

                lat, lon = locations[place]

                # Small GPS-like noise
                lat += rng.normal(0, 0.002)
                lon += rng.normal(0, 0.002)

                if place.startswith("Unknown"):
                    speed = rng.uniform(35, 90)
                    transport = rng.choice(["Car", "Bus", "Bike"])
                elif place in ["Home", "College", "Office"]:
                    speed = rng.uniform(0, 12)
                    transport = rng.choice(["Walk", "Bike", "Bus"])
                else:
                    speed = rng.uniform(5, 45)
                    transport = rng.choice(["Walk", "Bike", "Bus", "Car"])

                timestamp = date + pd.Timedelta(hours=hour)

                records.append({
                    "person_id": person_id,
                    "timestamp": timestamp,
                    "latitude": lat,
                    "longitude": lon,
                    "speed": speed,
                    "location_type": place,
                    "transport_mode": transport,
                    "is_synthetic_anomaly": int(is_unusual)
                })

    df = pd.DataFrame(records)
    df = df.sort_values(["person_id", "timestamp"]).reset_index(drop=True)

    # Previous movement
    df["previous_latitude"] = df.groupby("person_id")["latitude"].shift(1)
    df["previous_longitude"] = df.groupby("person_id")["longitude"].shift(1)

    return df


# ---------------------------------------------------------------------
# Geographic feature engineering
# ---------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def preprocess_and_engineer(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])

    df = df.drop_duplicates()

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_name"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["previous_latitude"] = (
        df.groupby("person_id")["latitude"].shift(1)
    )
    df["previous_longitude"] = (
        df.groupby("person_id")["longitude"].shift(1)
    )

    df["distance_km"] = haversine_km(
        df["previous_latitude"].fillna(df["latitude"]),
        df["previous_longitude"].fillna(df["longitude"]),
        df["latitude"],
        df["longitude"]
    )

    df["time_diff_minutes"] = (
        df.groupby("person_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    # Avoid division by zero
    denominator = df["time_diff_minutes"].replace(0, np.nan)
    calculated_speed = df["distance_km"] / (denominator / 60)
    df["calculated_speed"] = calculated_speed.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(df["speed"])

    # Direction/bearing
    lat1 = np.radians(df["previous_latitude"].fillna(df["latitude"]))
    lon1 = np.radians(df["previous_longitude"].fillna(df["longitude"]))
    lat2 = np.radians(df["latitude"])
    lon2 = np.radians(df["longitude"])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    df["bearing"] = (
        np.degrees(np.arctan2(x, y)) + 360
    ) % 360

    # Synthetic city center reference
    center_lat = df["latitude"].median()
    center_lon = df["longitude"].median()

    df["distance_from_center_km"] = haversine_km(
        df["latitude"],
        df["longitude"],
        center_lat,
        center_lon
    )

    # Historical location frequency
    location_counts = df["location_type"].value_counts(normalize=True)
    df["historical_frequency"] = (
        df["location_type"].map(location_counts).fillna(0)
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------

def run_clustering(df, n_clusters=5):
    features = [
        "latitude",
        "longitude",
        "speed",
        "hour",
        "distance_km"
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    result = df.copy()
    result["movement_cluster"] = clusters

    score = None
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(X_scaled, clusters)

    return result, model, scaler, score


# ---------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------

def run_anomaly_detection(df):
    features = [
        "speed",
        "distance_km",
        "hour",
        "distance_from_center_km",
        "bearing"
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=RANDOM_STATE
    )

    labels = model.fit_predict(X_scaled)
    raw_scores = -model.decision_function(X_scaled)

    result = df.copy()
    result["anomaly"] = labels
    result["anomaly_score"] = raw_scores

    # Convert score to approximately 0-1
    minimum = result["anomaly_score"].min()
    maximum = result["anomaly_score"].max()

    if maximum > minimum:
        result["anomaly_score_normalized"] = (
            result["anomaly_score"] - minimum
        ) / (maximum - minimum)
    else:
        result["anomaly_score_normalized"] = 0.0

    return result, model, scaler


# ---------------------------------------------------------------------
# Location prediction
# ---------------------------------------------------------------------

def prepare_next_location_data(df):
    result = df.copy()

    result["previous_location"] = (
        result.groupby("person_id")["location_type"].shift(1)
    )

    result["next_location"] = (
        result.groupby("person_id")["location_type"].shift(-1)
    )

    result = result.dropna(subset=["previous_location", "next_location"])

    # Encode categorical variables using one-hot encoding
    X = result[
        [
            "hour",
            "day_of_week",
            "speed",
            "distance_km",
            "distance_from_center_km",
            "previous_location",
            "transport_mode"
        ]
    ].copy()

    X = pd.get_dummies(
        X,
        columns=["previous_location", "transport_mode"],
        dtype=int
    )

    y = result["next_location"]

    return result, X, y


def train_location_model(df):
    data, X, y = prepare_next_location_data(df)

    if len(data) < 20 or y.nunique() < 2:
        return None

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=15,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    }

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "training_data": data
    }


def predict_probable_locations(model_bundle, df, person_id):
    model = model_bundle["model"]
    training_data = model_bundle["training_data"]

    person_data = df[df["person_id"] == person_id].sort_values("timestamp")

    if person_data.empty:
        return pd.DataFrame()

    latest = person_data.iloc[-1]

    previous_location = latest["location_type"]
    transport_mode = latest["transport_mode"]

    input_df = pd.DataFrame([{
        "hour": latest["hour"],
        "day_of_week": latest["day_of_week"],
        "speed": latest["speed"],
        "distance_km": latest["distance_km"],
        "distance_from_center_km": latest["distance_from_center_km"],
        "previous_location": previous_location,
        "transport_mode": transport_mode
    }])

    input_df = pd.get_dummies(
        input_df,
        columns=["previous_location", "transport_mode"],
        dtype=int
    )

    # Match model columns
    expected_columns = model.feature_names_in_

    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    input_df = input_df[expected_columns]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    result = pd.DataFrame({
        "location": classes,
        "probability": probabilities
    })

    # Historical frequency
    frequency = (
        training_data["next_location"]
        .value_counts(normalize=True)
        .rename("historical_frequency")
    )

    result = result.merge(
        frequency,
        left_on="location",
        right_index=True,
        how="left"
    )

    result["historical_frequency"] = (
        result["historical_frequency"].fillna(0)
    )

    result["probability"] = result["probability"].fillna(0)

    return result.sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Route prediction
# ---------------------------------------------------------------------

def build_movement_graph(df):
    if not NETWORKX_OK:
        return None

    G = nx.DiGraph()

    transition_counts = (
        df.sort_values(["person_id", "timestamp"])
        .assign(
            next_location=lambda x:
            x.groupby("person_id")["location_type"].shift(-1)
        )
        .dropna(subset=["next_location"])
        .groupby(["location_type", "next_location"])
        .size()
        .reset_index(name="count")
    )

    for _, row in transition_counts.iterrows():
        source = row["location_type"]
        target = row["next_location"]

        # Higher historical count means a more preferred transition.
        # Convert it to a cost so shortest-path can favor common transitions.
        cost = 1.0 / float(row["count"])

        G.add_edge(
            source,
            target,
            weight=cost,
            frequency=int(row["count"])
        )

    return G


def predict_routes(G, source, destinations, max_routes=5):
    if G is None:
        return []

    routes = []

    for destination in destinations:
        try:
            if source == destination:
                continue

            path = nx.shortest_path(
                G,
                source=source,
                target=destination,
                weight="weight"
            )

            cost = nx.path_weight(G, path, weight="weight")

            routes.append({
                "destination": destination,
                "route": " → ".join(path),
                "cost": cost
            })

        except Exception:
            continue

    routes = sorted(routes, key=lambda x: x["cost"])
    return routes[:max_routes]


# ---------------------------------------------------------------------
# Search-priority scoring
# ---------------------------------------------------------------------

def calculate_priority_scores(location_predictions, df):
    result = location_predictions.copy()

    # Recency: synthetic latest observation is treated as most relevant.
    max_timestamp = df["timestamp"].max()

    last_seen = (
        df.groupby("location_type")["timestamp"]
        .max()
        .rename("last_seen")
    )

    result = result.merge(
        last_seen,
        left_on="location",
        right_index=True,
        how="left"
    )

    hours_since = (
        max_timestamp - result["last_seen"]
    ).dt.total_seconds() / 3600

    result["recency_score"] = np.exp(
        -np.maximum(hours_since.fillna(999), 0) / 24
    )

    # Synthetic anomaly association
    anomaly_by_location = (
        df.groupby("location_type")["anomaly_score_normalized"]
        .mean()
        .rename("anomaly_score")
    )

    result = result.merge(
        anomaly_by_location,
        left_on="location",
        right_index=True,
        how="left"
    )

    result["anomaly_score"] = result["anomaly_score"].fillna(0)

    result["historical_frequency"] = (
        result["historical_frequency"].fillna(0)
    )

    # Transparent weighted score
    result["priority_score"] = (
        0.45 * result["probability"]
        + 0.25 * result["historical_frequency"]
        + 0.20 * result["anomaly_score"]
        + 0.10 * result["recency_score"]
    ) * 100

    return result.sort_values(
        "priority_score",
        ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------

def get_feature_importance(model_bundle):
    model = model_bundle["model"]

    importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_
    })

    return importance.sort_values(
        "importance",
        ascending=False
    )


# ---------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------

def create_map(df, priority_df=None, person_id=None):
    if not FOLIUM_OK:
        return None

    map_df = df.copy()

    if person_id:
        map_df = map_df[map_df["person_id"] == person_id]

    if map_df.empty:
        return None

    center = [
        map_df["latitude"].mean(),
        map_df["longitude"].mean()
    ]

    m = folium.Map(
        location=center,
        zoom_start=12,
        control_scale=True
    )

    # Historical path
    points = map_df.sort_values("timestamp")[
        ["latitude", "longitude"]
    ].values.tolist()

    if len(points) > 1:
        folium.PolyLine(
            points,
            tooltip="Historical movement"
        ).add_to(m)

    # Movement points
    for _, row in map_df.tail(250).iterrows():
        popup = (
            f"Person: {row['person_id']}<br>"
            f"Time: {row['timestamp']}<br>"
            f"Location: {row['location_type']}<br>"
            f"Speed: {row['speed']:.1f}<br>"
            f"Anomaly: {row.get('anomaly', 1)}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            popup=popup,
            fill=True
        ).add_to(m)

    # Highlight synthetic anomalies
    if "anomaly" in map_df.columns:
        anomalies = map_df[map_df["anomaly"] == -1]

        for _, row in anomalies.tail(100).iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=f"Synthetic anomaly: {row['location_type']}",
                icon=folium.Icon(icon="warning")
            ).add_to(m)

    # Add high-priority candidate locations
    if priority_df is not None:
        locations = (
            df.groupby("location_type")[["latitude", "longitude"]]
            .mean()
        )

        for _, row in priority_df.head(5).iterrows():
            loc = row["location"]

            if loc in locations.index:
                lat = locations.loc[loc, "latitude"]
                lon = locations.loc[loc, "longitude"]

                folium.Marker(
                    location=[lat, lon],
                    popup=(
                        f"Candidate: {loc}<br>"
                        f"Priority: {row['priority_score']:.1f}"
                    ),
                    icon=folium.Icon(icon="star")
                ).add_to(m)

    return m


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

def plot_hourly_movement(df):
    hourly = df.groupby("hour").size()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of movements")
    ax.set_title("Movement Frequency by Hour")
    ax.grid(alpha=0.25)
    return fig


def plot_location_frequency(df):
    counts = df["location_type"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Location")
    ax.set_ylabel("Visits")
    ax.set_title("Most Frequently Visited Locations")
    ax.tick_params(axis="x", rotation=35)
    return fig


def plot_anomalies(df):
    counts = df["anomaly"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        ["Normal", "Anomaly"],
        [counts.get(1, 0), counts.get(-1, 0)]
    )
    ax.set_ylabel("Records")
    ax.set_title("Anomaly Detection Summary")
    return fig


# ---------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "synthetic_movement_data.csv")

    if os.path.exists(path):
        raw = pd.read_csv(path)
    else:
        raw = generate_synthetic_dataset()
        raw.to_csv(path, index=False)

    return preprocess_and_engineer(raw)


@st.cache_data
def cached_clustering(df, n_clusters):
    result, _, _, score = run_clustering(df, n_clusters)
    return result, score


@st.cache_data
def cached_anomaly(df):
    result, _, _ = run_anomaly_detection(df)
    return result


def main():
    st.title("🔎 CASEFILE")
    st.subheader(
        "AI-Powered Missing Person Investigation and "
        "Probable Location Prediction System"
    )

    st.info(
        "Academic simulation only. The system uses fictional/synthetic "
        "movement information and provides investigation-support scores, "
        "not factual location claims about real people."
    )

    # Sidebar
    st.sidebar.header("⚙️ System Controls")

    if st.sidebar.button("Generate New Synthetic Dataset"):
        if os.path.exists(
            os.path.join(DATA_DIR, "synthetic_movement_data.csv")
        ):
            os.remove(
                os.path.join(DATA_DIR, "synthetic_movement_data.csv")
            )
        st.cache_data.clear()
        st.rerun()

    df = load_data()

    people = sorted(df["person_id"].unique())
    selected_person = st.sidebar.selectbox(
        "Select fictional case/person",
        people
    )

    n_clusters = st.sidebar.slider(
        "Movement clusters",
        min_value=2,
        max_value=10,
        value=5
    )

    # ML
    clustered_df, cluster_score = cached_clustering(
        df,
        n_clusters
    )

    analyzed_df = cached_anomaly(clustered_df)

    location_bundle = train_location_model(analyzed_df)

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Movement Records",
        f"{len(analyzed_df):,}"
    )

    c2.metric(
        "Fictional Cases",
        df["person_id"].nunique()
    )

    c3.metric(
        "Detected Anomalies",
        int((analyzed_df["anomaly"] == -1).sum())
    )

    c4.metric(
        "Clusters",
        n_clusters
    )

    # ---------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------

    (
        tab_overview,
        tab_eda,
        tab_cluster,
        tab_anomaly,
        tab_prediction,
        tab_routes,
        tab_explain,
        tab_data
    ) = st.tabs([
        "📊 Overview",
        "📈 EDA",
        "🧩 Clustering",
        "🚨 Anomaly Detection",
        "📍 Location Prediction",
        "🛣️ Route Prediction",
        "💡 Explainable AI",
        "🗃️ Data"
    ])

    # ---------------------------------------------------------------
    # Overview
    # ---------------------------------------------------------------

    with tab_overview:
        st.markdown("### Investigation Workflow")

        st.markdown("""
        ```text
        Synthetic Data
              ↓
        Preprocessing
              ↓
        Feature Engineering
              ↓
        Exploratory Data Analysis
              ↓
        Movement Clustering
              ↓
        Anomaly Detection
              ↓
        Location Prediction
              ↓
        Route Prediction
              ↓
        Search-Priority Scoring
              ↓
        Explainable AI
              ↓
        Interactive Map
              ↓
        Streamlit Dashboard
        ```
        """)

        st.markdown("### Selected fictional case")

        person_df = analyzed_df[
            analyzed_df["person_id"] == selected_person
        ]

        if not person_df.empty:
            latest = person_df.sort_values("timestamp").iloc[-1]

            a, b, c = st.columns(3)

            a.metric(
                "Latest synthetic location",
                latest["location_type"]
            )

            b.metric(
                "Latest speed",
                f"{latest['speed']:.1f}"
            )

            c.metric(
                "Latest anomaly score",
                f"{latest['anomaly_score_normalized']:.2f}"
            )

        if FOLIUM_OK:
            st.markdown("### Interactive movement map")

            map_obj = create_map(
                analyzed_df,
                person_id=selected_person
            )

            if map_obj:
                st_folium(
                    map_obj,
                    width=None,
                    height=550
                )
        else:
            st.warning(
                "Install folium and streamlit-folium to enable maps."
            )

    # ---------------------------------------------------------------
    # EDA
    # ---------------------------------------------------------------

    with tab_eda:
        st.header("Exploratory Data Analysis")

        st.pyplot(
            plot_hourly_movement(analyzed_df),
            clear_figure=True
        )

        st.pyplot(
            plot_location_frequency(analyzed_df),
            clear_figure=True
        )

        st.markdown("### Summary statistics")

        st.dataframe(
            analyzed_df[
                [
                    "speed",
                    "distance_km",
                    "distance_from_center_km",
                    "bearing"
                ]
            ].describe().round(2),
            use_container_width=True
        )

    # ---------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------

    with tab_cluster:
        st.header("Movement Pattern Clustering")

        st.write(
            "K-Means groups movements according to geographic and "
            "temporal/movement characteristics."
        )

        if cluster_score is not None:
            st.metric(
                "Silhouette Score",
                f"{cluster_score:.3f}"
            )

        cluster_summary = (
            clustered_df.groupby("movement_cluster")
            .agg(
                records=("movement_cluster", "size"),
                average_speed=("speed", "mean"),
                average_distance=("distance_km", "mean"),
                average_hour=("hour", "mean")
            )
            .round(2)
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        st.markdown("### Cluster map")

        if FOLIUM_OK:
            cluster_map = folium.Map(
                location=[
                    clustered_df["latitude"].mean(),
                    clustered_df["longitude"].mean()
                ],
                zoom_start=12
            )

            for _, row in clustered_df.sample(
                min(1000, len(clustered_df)),
                random_state=RANDOM_STATE
            ).iterrows():

                folium.CircleMarker(
                    location=[
                        row["latitude"],
                        row["longitude"]
                    ],
                    radius=3,
                    tooltip=(
                        f"Cluster: {row['movement_cluster']}"
                    )
                ).add_to(cluster_map)

            st_folium(
                cluster_map,
                width=None,
                height=500
            )

    # ---------------------------------------------------------------
    # Anomaly
    # ---------------------------------------------------------------

    with tab_anomaly:
        st.header("Anomaly Detection")

        st.write(
            "Isolation Forest identifies movement observations that "
            "differ substantially from the learned normal movement pattern."
        )

        st.pyplot(
            plot_anomalies(analyzed_df),
            clear_figure=True
        )

        anomalies = analyzed_df[
            analyzed_df["anomaly"] == -1
        ].sort_values(
            "anomaly_score_normalized",
            ascending=False
        )

        st.markdown("### Highest-scoring synthetic anomalies")

        st.dataframe(
            anomalies[
                [
                    "person_id",
                    "timestamp",
                    "location_type",
                    "speed",
                    "distance_km",
                    "anomaly_score_normalized"
                ]
            ].head(30).round(3),
            use_container_width=True
        )

    # ---------------------------------------------------------------
    # Location prediction
    # ---------------------------------------------------------------

    with tab_prediction:
        st.header("Probable Location Prediction")

        if location_bundle is None:
            st.error(
                "Not enough data to train the location prediction model."
            )
        else:
            metrics = location_bundle["metrics"]

            a, b, c, d = st.columns(4)

            a.metric(
                "Accuracy",
                f"{metrics['accuracy']:.3f}"
            )
            b.metric(
                "Precision",
                f"{metrics['precision']:.3f}"
            )
            c.metric(
                "Recall",
                f"{metrics['recall']:.3f}"
            )
            d.metric(
                "F1",
                f"{metrics['f1']:.3f}"
            )

            predictions = predict_probable_locations(
                location_bundle,
                analyzed_df,
                selected_person
            )

            if not predictions.empty:
                priority = calculate_priority_scores(
                    predictions,
                    analyzed_df
                )

                st.markdown(
                    "### Candidate locations for the selected "
                    "fictional case"
                )

                display = priority[
                    [
                        "location",
                        "probability",
                        "historical_frequency",
                        "anomaly_score",
                        "recency_score",
                        "priority_score"
                    ]
                ].copy()

                display.columns = [
                    "Location",
                    "Model Probability",
                    "Historical Frequency",
                    "Anomaly Score",
                    "Recency Score",
                    "Search Priority"
                ]

                st.dataframe(
                    display.head(10).style.format({
                        "Model Probability": "{:.3f}",
                        "Historical Frequency": "{:.3f}",
                        "Anomaly Score": "{:.3f}",
                        "Recency Score": "{:.3f}",
                        "Search Priority": "{:.2f}"
                    }),
                    use_container_width=True
                )

                st.caption(
                    "Scores are synthetic academic outputs and should "
                    "not be interpreted as certainty or real-world tracking."
                )

                # Map candidate locations
                if FOLIUM_OK:
                    candidate_map = create_map(
                        analyzed_df,
                        priority_df=priority,
                        person_id=selected_person
                    )

                    if candidate_map:
                        st_folium(
                            candidate_map,
                            width=None,
                            height=550
                        )

    # ---------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------

    with tab_routes:
        st.header("Route Prediction")

        if not NETWORKX_OK:
            st.warning(
                "Install NetworkX to enable graph-based route prediction."
            )
        else:
            graph = build_movement_graph(analyzed_df)

            person_df = analyzed_df[
                analyzed_df["person_id"] == selected_person
            ].sort_values("timestamp")

            if not person_df.empty:
                source = person_df.iloc[-1]["location_type"]

                destinations = [
                    x for x in analyzed_df["location_type"].unique()
                    if x != source
                ]

                routes = predict_routes(
                    graph,
                    source,
                    destinations
                )

                st.write(
                    f"Current synthetic source location: **{source}**"
                )

                if routes:
                    route_df = pd.DataFrame(routes)

                    st.dataframe(
                        route_df,
                        use_container_width=True
                    )

                    st.markdown(
                        "Routes are generated from historical transition "
                        "patterns represented as a graph."
                    )
                else:
                    st.info(
                        "No graph route could be generated for this case."
                    )

    # ---------------------------------------------------------------
    # Explainability
    # ---------------------------------------------------------------

    with tab_explain:
        st.header("Explainable AI")

        st.write(
            "The Random Forest model exposes feature importance. "
            "This helps demonstrate which input variables contribute "
            "to the learned decision process."
        )

        if location_bundle is not None:
            importance = get_feature_importance(location_bundle)

            st.dataframe(
                importance.head(20).round(5),
                use_container_width=True
            )

            top = importance.head(10).sort_values("importance")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(
                top["feature"],
                top["importance"]
            )
            ax.set_xlabel("Importance")
            ax.set_title(
                "Location Prediction Feature Importance"
            )

            st.pyplot(
                fig,
                clear_figure=True
            )

            st.markdown("### Interpretation")

            if not importance.empty:
                features_text = ", ".join(
                    importance.head(5)["feature"].tolist()
                )

                st.info(
                    "The five highest-importance model features in this "
                    f"synthetic run are: {features_text}. Feature importance "
                    "describes model behavior; it does not establish a "
                    "causal relationship."
                )

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------

    with tab_data:
        st.header("Processed Dataset")

        st.dataframe(
            analyzed_df.head(500),
            use_container_width=True,
            height=500
        )

        csv = analyzed_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download processed dataset",
            data=csv,
            file_name="casefile_processed_movement_data.csv",
            mime="text/csv"
        )

        st.markdown("### Feature list")

        features = [
            "person_id",
            "timestamp",
            "latitude",
            "longitude",
            "speed",
            "location_type",
            "transport_mode",
            "hour",
            "day_of_week",
            "is_weekend",
            "distance_km",
            "time_diff_minutes",
            "calculated_speed",
            "bearing",
            "distance_from_center_km",
            "historical_frequency",
            "movement_cluster",
            "anomaly",
            "anomaly_score",
            "anomaly_score_normalized"
        ]

        st.code("\n".join(features))

    # Footer
    st.markdown("---")
    st.caption(
        "CASEFILE — Academic Advanced ML simulation. "
        "Synthetic data only. Human investigators/authorized professionals "
        "would remain responsible for any real-world decisions."
    )


if __name__ == "__main__":
    main()
