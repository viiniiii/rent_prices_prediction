import os
import datetime
import base64
import json

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="Berlin Rent Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    /* Brand palette -- grounded in Berlin's brick Altbau facades and the Spree at dusk */
    :root {
        --ink:        #1E2328;
        --brick:      #C1592E;
        --brick-dark: #8F3E1F;
        --slate:      #4C6570;
        --mist:       #EEF2F1;
        --warn:       #B23B3B;
        --muted:      #64767B;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--ink);
        color: #fff;
    }
    section[data-testid="stSidebar"] * { color: #fff !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, var(--ink) 0%, var(--brick-dark) 100%);
        color: white;
        padding: 2.8rem 2.4rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.3rem; font-weight: 700; margin: 0 0 0.5rem 0; }
    .hero p  { font-size: 1.08rem; margin: 0; opacity: 0.9; max-width: 640px; line-height: 1.5; }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap; }
    .metric-card {
        flex: 1;
        min-width: 150px;
        background: #fff;
        border-left: 4px solid var(--brick);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .val { font-size: 1.8rem; font-weight: 700; color: var(--brick); font-family: 'Fraunces', serif; }
    .metric-card .lbl { font-size: 0.8rem; color: var(--muted); margin-top: 0.15rem; }

    /* Info / warning / disclaimer boxes */
    .info-box {
        background: var(--mist);
        border-left: 4px solid var(--slate);
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        color: var(--ink);
    }
    .disclaimer {
        background: #FBEFE9;
        border-left: 6px solid var(--warn);
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #6E2A20;
        line-height: 1.6;
    }

    /* Section headers */
    h2 { color: var(--ink); border-bottom: 2px solid var(--brick); padding-bottom: 0.3rem; }
    h3 { color: var(--brick-dark); }

    .coming-soon {
        background: #fff;
        border: 1px dashed var(--slate);
        border-radius: 10px;
        padding: 2.5rem;
        text-align: center;
        color: var(--muted);
        margin-top: 1rem;
    }

    /* Prediction result */
    .result {
        background: var(--mist);
        border: 2px solid var(--brick);
        border-radius: 10px;
        padding: 1.8rem;
        text-align: center;
        margin: 1.2rem 0;
    }
    .result .amount { font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 700; color: var(--brick); }
    .result .range { font-size: 1rem; color: var(--muted); margin-top: 0.3rem; }
    .result .confidence { font-size: 1.05rem; color: var(--slate); margin-top: 0.7rem; font-weight: 600; }
    .result .confidence.low { color: var(--warn); }

    div[data-testid="stForm"] {
        background: #fff;
        border: 1px solid #E3E7E6;
        border-radius: 10px;
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
    }

    /* Input borders + focus rings pick up the brick accent
       (primaryColor in .streamlit/config.toml handles buttons,
       checkboxes, and radio dots automatically) */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        border-color: #D8DEDD !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: var(--brick) !important;
        box-shadow: 0 0 0 1px var(--brick) !important;
    }

    /* Form labels + "Basics"/"Amenities"/"Building details" subheaders in brick */
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stForm"] .stMarkdown p,
    div[data-testid="stForm"] .stMarkdown strong {
        color: var(--brick) !important;
    }
</style>
""", unsafe_allow_html=True)

N_LISTINGS = 15588
N_FEATURES = 13
TEST_R2 = 0.92
TEST_RMSE = 204

# Final model's exact evaluation numbers.
FINAL_TEST_R2 = 0.9198
FINAL_TEST_RMSE = 204.6
FINAL_TEST_MAE = 124.7
FINAL_TRAIN_R2 = 0.9834
FINAL_CV_RMSE = 234.2

XGB_TEST_R2 = 0.9398       # XGBoost, without 7 least-important features -- best raw R², but no native uncertainty
XGB_QUANTILE_TEST_R2 = 0.8870  # XGBoost Quantile Regression (50th pct) -- adds uncertainty, costs accuracy

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

SECTIONS = ["Overview", "Explore the Data", "Model & Performance", "Model Card", "Predict Your Rent"]

def render_overview():
    st.markdown("""
    <div class="hero">
    <h1>What's a fair cold rent in Berlin?</h1>
    <p>A transparent tool that estimates a Berlin apartment's <em>Kaltmiete</em> (cold rent,
    excluding utilities) from its features, and shows exactly which of those features
    are pushing the price up or down.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
    <div class="metric-card"><div class="val">{:,}</div><div class="lbl">Berlin listings analyzed</div></div>
    <div class="metric-card"><div class="val">{}</div><div class="lbl">Property features used</div></div>
    <div class="metric-card"><div class="val">{:.0%}</div><div class="lbl">Variance explained (R²) on unseen listings</div></div>
    <div class="metric-card"><div class="val">±€{}</div><div class="lbl">Typical prediction error (RMSE)</div></div>
    </div>
    """.format(N_LISTINGS, N_FEATURES, TEST_R2, TEST_RMSE), unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    <strong>This is an estimate, not a valuation.</strong> The model was trained on historical
    Berlin listings scraped from Immowelt. It reflects patterns in that data, not a
    professional appraisal, and can be wrong for unusual properties or districts the
    model has seen very few of.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What this tool does")
        st.markdown("""
        - Estimates the **cold rent** for an apartment from details like area, rooms,
          district, floor, and amenities
        - Shows a **confidence range**, not just a single number — some listings are
          much easier to predict than others
        - Explains **which features drove the estimate** up or down for that specific apartment
        - Lets you explore how the model performs **across districts and property types**
        """)
    with col2:
        st.markdown("### Who is this for")
        st.markdown("""
        - **Renters** who want a sense of whether an asking price looks reasonable
        - **Landlords** curious how their listing compares to similar apartments
        - Anyone curious what actually moves rent prices in Berlin — area and
          room count matter most, but district and move-in availability matter too
        """)

    st.markdown("---")
    st.markdown("### About the data")
    st.markdown("""
    <div class="info-box">
    The dataset comes from real apartment listings posted on Immowelt. After cleaning
    (removing outliers, merging rare categories, dropping a fully-constant column),
    {:,} listings remained. The final model is a <strong>Random Forest</strong> trained
    on the 13 features that mattered most — a handful of amenities like garden and
    parking access were tested but dropped because they added no predictive value.
    </div>
    """.format(N_LISTINGS), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Glossary")
    st.markdown("""
    <div class="info-box">
    A few terms that come up throughout the tool, explained in plain language.
    </div>
    """, unsafe_allow_html=True)

    glossary = [
        ("Cold rent (Kaltmiete)",
         "The base rent for an apartment, before utilities like heating, water, and building maintenance (Nebenkosten) are added on top. This is what the model predicts."),
        ("Random Forest",
         "The type of model used here. It works by training many decision trees on random subsets of the data, then averaging their individual predictions. Because each tree votes independently, we can also see how much the trees agree — which tells us how confident the model is."),
        ("R² (R-squared)",
         "A score from 0 to 1 showing how much of the variation in rent prices the model explains. An R² of 0.92 means the model captures 92% of what makes one apartment's rent differ from another's."),
        ("RMSE (Root Mean Squared Error)",
         "The typical size of the model's prediction error, in euros. It penalizes large misses more than small ones, so it's a good read on how far off a 'bad' prediction tends to be."),
        ("Feature importance",
         "A ranking of which apartment characteristics most influence the model's predictions overall, across all listings — not for one specific apartment."),
        ("Prediction confidence / uncertainty",
         "Because a Random Forest is made of many trees, each apartment gets many individual predictions. If the trees mostly agree, the model is confident. If they're spread out, the estimate is less reliable — and the tool will show you that."),
    ]
    for term, definition in glossary:
        with st.expander(f"**{term}**"):
            st.markdown(definition)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "rent_listings_final.csv")
BERLIN_PLZ_SHAPEFILE_URL = (
    "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/"
    "master/berlin_postleitzahlen.zip"
)

CATEGORY_EXPLORE_OPTIONS = [
    "free_from", "elevator", "has_built-in_kitchen", "has_shower", "has_bathtub",
    "has_balkon", "has_terrasse", "has_garten", "parking", "has_basement",
    "is_barrier-free", "flooring_type", "energy_source", "heating_type",
    "property_condition",
]


@st.cache_data(show_spinner="Loading listings...")
def load_full_dataset(data_path):
    from rent_pipeline import load_data
    return load_data(data_path)


@st.cache_data(show_spinner="Downloading Berlin postal code boundaries...")
def load_berlin_plz_shapes():
    """Same source as rent_prices_consumption.ipynb, cell 29. Downloaded once
    and cached for the life of the app process, not re-fetched per page view."""
    import tempfile
    import zipfile
    import io
    import requests
    import geopandas as gpd

    r = requests.get(BERLIN_PLZ_SHAPEFILE_URL, timeout=30)
    r.raise_for_status()
    with tempfile.TemporaryDirectory() as tmpdir:
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(tmpdir)
        gdf = gpd.read_file(f"{tmpdir}/berlin_postleitzahlen.shp")
    gdf["PLZ99"] = gdf["PLZ99"].astype(str)
    return gdf


def render_choropleth_section(df):
    st.markdown("### Interactive map: price per m² by postal code")

    try:
        gdf = load_berlin_plz_shapes()
    except ImportError:
        st.info(
            "This map needs the `geopandas` and `requests` packages "
            "(`pip install geopandas requests`) — skipping it for now."
        )
        return
    except Exception:
        st.info(
            "Couldn't download the Berlin postal code boundaries (needs internet "
            "access) — skipping the map for now."
        )
        return

    df_map = df.dropna(subset=["zip_code", "area", "cold_rent"]).copy()
    df_map = df_map[df_map["area"] > 0]
    df_map["PLZ99"] = df_map["zip_code"].astype(str)
    df_map["price_per_m2"] = df_map["cold_rent"] / df_map["area"]

    zip_stats = (
        df_map.groupby("PLZ99")["price_per_m2"]
        .agg(median_ppm2="median", n_listings="count")
        .reset_index()
    )
    merged = gdf.merge(zip_stats, on="PLZ99", how="left")
    geojson = json.loads(merged.to_json())

    lo, hi = merged["median_ppm2"].quantile([0.05, 0.95])
    fig_map = px.choropleth_map(
        merged, geojson=geojson, locations="PLZ99", color="median_ppm2",
        featureidkey="properties.PLZ99",
        color_continuous_scale="OrRd",
        range_color=(lo, hi),
        map_style="carto-positron",
        center={"lat": 52.52, "lon": 13.405},
        zoom=9.3,
        opacity=0.75,
        hover_data={"PLZ99": True, "median_ppm2": ":.1f", "n_listings": True},
        labels={"median_ppm2": "Median €/m²", "n_listings": "Listings"},
    )
    fig_map.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, height=520)
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "Darker = more expensive per square meter. Central postal codes (Mitte, "
        "Prenzlauer Berg, Charlottenburg) stand out clearly, alongside a few pricey "
        "pockets further out."
    )

    zip_options = ["All Berlin"] + sorted(zip_stats["PLZ99"].tolist())
    selected_zip = st.selectbox("Zoom in on a postal code", zip_options)
    subset = df_map if selected_zip == "All Berlin" else df_map[df_map["PLZ99"] == selected_zip]
    label = "Berlin-wide" if selected_zip == "All Berlin" else f"PLZ {selected_zip}"

    col_a, col_b = st.columns(2)
    with col_a:
        fig_kde = px.histogram(
            subset, x="price_per_m2", nbins=30, color_discrete_sequence=["#C1592E"],
            title=f"€/m² distribution — {label}",
            labels={"price_per_m2": "€/m²"},
        )
        st.plotly_chart(fig_kde, use_container_width=True)
    with col_b:
        fig_zip_scatter = px.scatter(
            subset, x="area", y="cold_rent", opacity=0.6,
            color_discrete_sequence=["#4C6570"],
            title=f"Area vs. cold rent — {label} ({len(subset)} listings)",
            labels={"area": "Area (m²)", "cold_rent": "Cold rent (€)"},
        )
        st.plotly_chart(fig_zip_scatter, use_container_width=True)


def render_category_explorer(df):
    st.markdown("### Explore by feature")
    st.markdown(
        "Pick an amenity or category to see how it splits the area-vs-rent relationship. "
        "Clearly separated colors mean the feature is strongly associated with price; "
        "heavy overlap means it isn't telling you much on its own."
    )
    available_cats = [c for c in CATEGORY_EXPLORE_OPTIONS if c in df.columns]
    feature = st.selectbox("Category", available_cats)

    sample = df.dropna(subset=["area", "cold_rent", feature])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=1)

    fig = px.scatter(
        sample, x="area", y="cold_rent", color=sample[feature].astype(str),
        opacity=0.55,
        title=f"Cold rent vs. area, colored by {feature}",
        labels={"area": "Area (m²)", "cold_rent": "Cold rent (€)", "color": feature},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_explore_page():
    st.markdown("## Explore the Data")
    st.markdown("""
    <div class="info-box">
    A look at the 15,588 Berlin listings behind the model, before any of it gets
    turned into a prediction.
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_full_dataset(DATA_FILE)
    except FileNotFoundError:
        st.error(
            f"Couldn't find the dataset at `{DATA_FILE}`. Run this app from the "
            "same folder the notebooks expect."
        )
        st.stop()

    median_rent = df["cold_rent"].median()
    mean_rent = df["cold_rent"].mean()
    n_districts = df["district"].nunique()

    st.markdown(f"""
    <div class="metric-row">
    <div class="metric-card"><div class="val">{len(df):,}</div><div class="lbl">Listings</div></div>
    <div class="metric-card"><div class="val">€{median_rent:,.0f}</div><div class="lbl">Median cold rent</div></div>
    <div class="metric-card"><div class="val">€{mean_rent:,.0f}</div><div class="lbl">Average cold rent</div></div>
    <div class="metric-card"><div class="val">{n_districts}</div><div class="lbl">Districts covered</div></div>
    </div>
    """, unsafe_allow_html=True)

    render_choropleth_section(df)

    st.markdown("### Rent distribution")
    fig_hist = px.histogram(
        df, x="cold_rent", nbins=60, color_discrete_sequence=["#C1592E"],
        title="Distribution of cold rent across all listings",
        labels={"cold_rent": "Cold rent (€)"},
    )
    fig_hist.add_vline(x=median_rent, line_dash="dash", line_color="#4C6570", annotation_text="median")
    fig_hist.update_layout(yaxis_title="Listings")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"Right-skewed, as rents usually are: most listings sit well below the average "
        f"(€{mean_rent:,.0f}), with a long tail of expensive outliers pulling the mean up "
        f"past the median (€{median_rent:,.0f})."
    )

    st.markdown("### Area vs. cold rent")
    sample = df.dropna(subset=["area", "cold_rent", "rooms"])
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=1)
    fig_scatter = px.scatter(
        sample, x="area", y="cold_rent", color="rooms",
        color_continuous_scale="Oranges", opacity=0.5,
        title="Cold rent vs. living area",
        labels={"area": "Area (m²)", "cold_rent": "Cold rent (€)", "rooms": "Rooms"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Area is the strongest single driver of price, close to a straight line. Room "
        "count tracks the same relationship rather than adding much of an independent effect."
    )

    render_category_explorer(df)

    st.markdown("### Price per m² by district")
    df_ppm = df.dropna(subset=["district", "area", "cold_rent"]).copy()
    df_ppm = df_ppm[df_ppm["area"] > 0]
    df_ppm["price_per_m2"] = df_ppm["cold_rent"] / df_ppm["area"]
    district_median = df_ppm.groupby("district")["price_per_m2"].median().sort_values(ascending=False)
    overall_median_ppm = df_ppm["price_per_m2"].median()
    top_n = 12
    top = district_median.head(top_n).sort_values(ascending=True)
    bottom = district_median.tail(top_n).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_top = px.bar(
            x=top.values, y=top.index, orientation="h",
            color_discrete_sequence=["#C1592E"],
            title=f"{top_n} most expensive districts",
            labels={"x": "Median €/m²", "y": ""},
        )
        fig_top.add_vline(x=overall_median_ppm, line_dash="dash", line_color="#4C6570")
        st.plotly_chart(fig_top, use_container_width=True)
    with col2:
        fig_bottom = px.bar(
            x=bottom.values, y=bottom.index, orientation="h",
            color_discrete_sequence=["#4C6570"],
            title=f"{top_n} most affordable districts",
            labels={"x": "Median €/m²", "y": ""},
        )
        fig_bottom.add_vline(x=overall_median_ppm, line_dash="dash", line_color="#C1592E")
        st.plotly_chart(fig_bottom, use_container_width=True)
    st.caption(
        f"Dashed line marks the citywide median (€{overall_median_ppm:.1f}/m²). Price per "
        "square meter controls for apartment size, so it's a cleaner way to compare "
        "districts than raw rent — some of the priciest-per-m² districts are far from "
        "central Berlin, likely because they're small, newer-built apartments."
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### When were these apartments built?")
        yearly = df.dropna(subset=["year_built"]).sort_values("year_built")
        yearly_count = yearly.groupby("year_built").size().reset_index(name="count")
        yearly_count["cumulative"] = yearly_count["count"].cumsum()
        fig_year = px.line(
            yearly_count, x="year_built", y="cumulative",
            title="Cumulative listings by construction year",
            labels={"year_built": "Year built", "cumulative": "Cumulative listings"},
        )
        fig_year.update_traces(line_color="#C1592E")
        st.plotly_chart(fig_year, use_container_width=True)
        st.caption(
            "Slow, steady growth through the World Wars, then a sharp acceleration "
            "after 2010 as new construction picks up."
        )
    with col4:
        st.markdown("### Which floor are apartments on?")
        floor_counts = df["floor"].dropna().value_counts().sort_index()
        fig_floor = px.bar(
            x=floor_counts.index.astype(str), y=floor_counts.values,
            color_discrete_sequence=["#C1592E"],
            title="Listings by floor",
            labels={"x": "Floor", "y": "Listings"},
        )
        st.plotly_chart(fig_floor, use_container_width=True)
        st.caption(
            "Most apartments sit on the 1st through 4th floors. Anything above the "
            "10th floor is grouped together as '11'."
        )


# Model loading + inference helpers

MODEL_PATH = os.path.join(BASE_DIR, "final_model.joblib")
MODEL_CARD_PATH = os.path.join(BASE_DIR, "model_card.pdf")


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


@st.cache_resource
def build_explainer(_pipeline):
    model = _pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


def predict_with_uncertainty(pipeline, X):
    """Same logic as the notebook: run every tree in the forest individually
    so we can report agreement (uncertainty), not just the averaged prediction."""
    preprocessor = pipeline.named_steps["preprocessor"]
    forest = pipeline.named_steps["model"]
    X_proc = preprocessor.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    tree_preds = np.stack([t.predict(X_proc) for t in forest.estimators_], axis=1)
    return tree_preds.mean(axis=1), tree_preds.std(axis=1), tree_preds


def get_categorical_options(pipeline):
    """Pull the exact category lists the encoder was fit on, straight from the
    saved pipeline, rather than hardcoding a guessed list that could drift
    out of sync with what the model actually accepts."""
    preprocessor = pipeline.named_steps["preprocessor"]
    options = {}
    for name, trans, cols in preprocessor.transformers_:
        if name != "cat":
            continue
        encoder = trans
        if hasattr(encoder, "named_steps"):
            encoder = list(encoder.named_steps.values())[-1]
        for col, cats in zip(cols, encoder.categories_):
            options[col] = list(cats)
    return options


@st.cache_data(show_spinner="Scoring the model on the held-out test set...")
def compute_test_results(_pipeline, data_path):
    """Same split as rent_prices_model_explainability_and_inference.ipynb
    (Part 1: Model exploration, cell 9): reload the raw data, redo the exact
    train/test split the model was evaluated on (same test_size and
    random_state), and score it. Cached so re-visiting the page doesn't
    re-run ~3,100 predictions every time."""
    from rent_pipeline import load_data, split_data

    df = load_data(data_path)
    _, X_test, _, y_test = split_data(df, test_size=0.2, random_state=16)

    y_pred = _pipeline.predict(X_test)

    results = X_test.copy().reset_index(drop=True)
    results["actual"] = y_test.values
    results["predicted"] = y_pred
    results["error"] = results["predicted"] - results["actual"]
    results["abs_error"] = results["error"].abs()
    return results


@st.cache_data(show_spinner=False)
def compute_feature_importance(_pipeline):
    """Mirrors the explainability notebook's Part 2, cell 35."""
    from rent_pipeline import get_feature_names

    importances = _pipeline.named_steps["model"].feature_importances_
    feature_names = get_feature_names(_pipeline)
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return fi_df.sort_values("importance", ascending=False)


def render_model_page():
    st.markdown("## Model & Performance")
    st.markdown("""
    <div class="info-box">
    How the final Random Forest actually performs on apartments it never saw during
    training, how it compares to the alternatives that were tried, and which
    features it leans on most.
    </div>
    """, unsafe_allow_html=True)

    try:
        bundle = load_bundle(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f"Couldn't find a trained model at `{MODEL_PATH}`. "
            "Run this app from the same folder the notebooks expect "
            "(`final_model.joblib` should be reachable at that relative path)."
        )
        st.stop()

    pipeline = bundle["pipeline"]

    st.markdown(f"""
    <div class="metric-row">
    <div class="metric-card"><div class="val">{FINAL_TEST_R2:.1%}</div><div class="lbl">Test R²</div></div>
    <div class="metric-card"><div class="val">±€{FINAL_TEST_RMSE:.0f}</div><div class="lbl">Test RMSE</div></div>
    <div class="metric-card"><div class="val">±€{FINAL_TEST_MAE:.0f}</div><div class="lbl">Test MAE</div></div>
    <div class="metric-card"><div class="val">{FINAL_TRAIN_R2:.1%}</div><div class="lbl">Train R² (for reference)</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    Train R² sitting above test R² ({FINAL_TRAIN_R2:.1%} vs {FINAL_TEST_R2:.1%}) is a normal
    amount of overfitting for a tuned Random Forest, not a red flag on its own — cross-validation
    RMSE (€{FINAL_CV_RMSE:.0f}) tracks closely with the test RMSE (€{FINAL_TEST_RMSE:.0f})
    rather than diverging from it, which is what you'd expect to see if it were memorizing
    the training set.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Why Random Forest, not XGBoost?")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        XGBoost actually scored higher on raw accuracy during model selection. But a Random
        Forest is an ensemble of independently-voting trees, so it can report *how much the
        trees disagree* as a built-in confidence signal — which is what powers the confidence
        estimate on the prediction page. A quantile-regression version of XGBoost was tried
        to get that same capability, but its accuracy dropped well below even the plain
        Random Forest, so the trade-off wasn't worth it.
        """)
    with col2:
        model_compare_df = pd.DataFrame({
            "model": ["XGBoost", "Random Forest\n(chosen)", "XGBoost\nQuantile"],
            "test_r2": [XGB_TEST_R2, FINAL_TEST_R2, XGB_QUANTILE_TEST_R2],
        })
        fig_cmp = px.bar(
            model_compare_df, x="model", y="test_r2",
            color="model",
            color_discrete_sequence=["#4C6570", "#C1592E", "#8F3E1F"],
            title="Test R² of the final candidates",
        )
        fig_cmp.update_layout(showlegend=False, yaxis_range=[0, 1], yaxis_title="Test R²", xaxis_title="")
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    results = compute_test_results(pipeline, DATA_FILE)

    st.markdown("### Predicted vs Actual")
    fig_pva = px.scatter(
        results, x="actual", y="predicted", color="abs_error",
        color_continuous_scale="RdYlGn_r",
        hover_data=["area", "rooms", "district", "floor", "heating_type"],
        title=f"Predicted vs actual cold rent — {len(results):,} test listings",
        labels={"actual": "Actual cold rent (€)", "predicted": "Predicted cold rent (€)", "abs_error": "Abs. error (€)"},
    )
    m = max(results["actual"].max(), results["predicted"].max())
    fig_pva.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig_pva, use_container_width=True)
    st.caption(
        "Points on the dashed line are exact predictions. The model tends to slightly "
        "underpredict the most expensive apartments — few enough of them in the data that "
        "the model can't fully learn what drives their price."
    )

    st.markdown("### Residuals")
    fig_hist = px.histogram(
        results, x="error", nbins=40, marginal="box",
        title="Distribution of prediction errors (predicted − actual)",
        color_discrete_sequence=["#C1592E"],
        labels={"error": "Error (€)"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Centered close to €0 with a roughly symmetric, bell-shaped spread — the model "
        "isn't systematically biased high or low across the test set as a whole."
    )

    st.markdown("### What drives the prediction most")
    fi_df = compute_feature_importance(pipeline)
    top_fi = fi_df.head(15).sort_values("importance")
    fig_fi = px.bar(
        top_fi, x="importance", y="feature", orientation="h",
        color_discrete_sequence=["#C1592E"],
        title="Top 15 feature importances",
        labels={"importance": "Importance", "feature": ""},
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(
        "Area and room count dominate, as expected. A few 'unknown' categories "
        "(missing availability, missing property condition) also carry real signal — "
        "listings that omit these details tend to differ systematically in price, "
        "not just at random."
    )


def pretty(value):
    return "Unknown / not specified" if str(value).lower() == "unknown" else str(value)


def render_model_card_page():
    st.markdown("## Model Card")
    st.markdown("""
    <div class="info-box">
    A short, standalone document covering what the model is, what it was trained on,
    how well it performs, and its known limitations — useful if you want to share or
    cite this tool without pointing someone at the whole codebase.
    </div>
    """, unsafe_allow_html=True)

    try:
        with open(MODEL_CARD_PATH, "rb") as f:
            pdf_bytes = f.read()
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{MODEL_CARD_PATH}`. It should sit in the same folder as "
            "this script."
        )
        st.stop()

    st.download_button(
        "Download model card (PDF)",
        data=pdf_bytes,
        file_name="berlin_rent_model_card.pdf",
        mime="application/pdf",
        type="primary",
    )

    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(f"""
    <iframe src="data:application/pdf;base64,{b64}" width="100%" height="850"
    style="border:1px solid #E3E7E6;border-radius:10px;margin-top:1rem;"></iframe>
    """, unsafe_allow_html=True)
    st.caption(
        "Some browsers block inline PDF previews in embedded frames — if the box above "
        "looks empty, the download button still works."
    )


FREE_FROM_LABELS = {
    "immediately": "Immediately",
    "not_immediately": "Not immediately (waiting period)",
    "unknown": "Not specified",
}

# The model was trained on raw floor codes ("-1", "0", "1" ... "11", where "11"
# groups everything from the 11th floor up). The dropdown needs human-friendly
# labels for the edge cases, but must map back to those exact codes before the
# listing is handed to the pipeline -- otherwise "Basement (-1)", "Ground floor
# (0)" and "11+" would be sent to the encoder as unrecognized category strings.
FLOOR_CODES = ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
FLOOR_CODE_LABELS = {
    "-1": "Basement (-1)",
    "0": "Ground floor (0)",
    "11": "11th floor and above",
}
FLOOR_DISPLAY_TO_CODE = {
    FLOOR_CODE_LABELS.get(code, code): code for code in FLOOR_CODES
}


def render_predict_page():
    st.markdown("## Predict Your Rent")
    st.markdown("""
    <div class="info-box">
    Fill in what you know about the apartment. Fields you leave as "unknown" are
    handled the same way the model was trained to handle missing data — it
    won't break the prediction, but more detail generally means a tighter
    confidence range.
    </div>
    """, unsafe_allow_html=True)

    try:
        bundle = load_bundle(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f"Couldn't find a trained model at `{MODEL_PATH}`. "
            "Run this app from the same folder the notebooks expect "
            "(`final_model.pkl` should be reachable at that relative path)."
        )
        st.stop()

    pipeline = bundle["pipeline"]
    cat_options = get_categorical_options(pipeline)
    current_year = datetime.date.today().year

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basics**")
            area = st.number_input("Living area (m²)", min_value=10.0, max_value=400.0, value=60.0, step=1.0)
            rooms = st.number_input("Number of rooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
            district = st.selectbox("District", sorted(cat_options.get("district", [])))
            free_from_label = st.selectbox("Availability", list(FREE_FROM_LABELS.values()))

            floor_label = st.selectbox(
                "Floor",
                ["Unknown"] + list(FLOOR_DISPLAY_TO_CODE.keys())
            )

            year_built = st.selectbox(
                "Construction year",
                ["Unknown"] + [str(year) for year in range(1850, current_year + 1)]
            )

        with col2:
            st.markdown("**Amenities**")
            elevator = st.checkbox("Elevator")
            has_kitchen = st.checkbox("Built-in kitchen (Einbauküche)")
            has_shower = st.checkbox("Shower")

            st.markdown("**Building details**")
            flooring_type = st.selectbox(
                "Flooring type", sorted(cat_options.get("flooring_type", [])), format_func=pretty
            )
            heating_type = st.selectbox(
                "Heating type", sorted(cat_options.get("heating_type", [])), format_func=pretty
            )
            energy_source = st.selectbox(
                "Energy source", sorted(cat_options.get("energy_source", [])), format_func=pretty
            )
            property_condition = st.selectbox(
                "Property condition", sorted(cat_options.get("property_condition", [])), format_func=pretty
            )

        submitted = st.form_submit_button("Estimate rent", type="primary", use_container_width=True)

    if not submitted:
        return

    free_from = [k for k, v in FREE_FROM_LABELS.items() if v == free_from_label][0]
    floor = None if floor_label == "Unknown" else FLOOR_DISPLAY_TO_CODE[floor_label]

    raw_listing = {
        "area": area,
        "rooms": rooms,
        "floor": floor,
        "free_from": free_from,
        "elevator": elevator,
        "has_built-in_kitchen": has_kitchen,
        "has_shower": has_shower,
        "flooring_type": flooring_type,
        "energy_source": energy_source,
        "heating_type": heating_type,
        "property_condition": property_condition,
        "year_built": None if year_built == "Unknown" else year_built,
        "district": district,
    }

    from rent_pipeline import validate_listing, get_feature_names

    new_data = validate_listing(raw_listing, bundle)
    _, _, tree_preds = predict_with_uncertainty(pipeline, new_data)
    tree_preds_flat = tree_preds[0]
    point_pred = float(np.mean(tree_preds_flat))

    within_10pct = np.abs(tree_preds_flat - point_pred) <= 0.10 * point_pred
    confidence = float(100 * within_10pct.mean())

    p10, p90 = np.percentile(tree_preds_flat, [10, 90])

    confidence_class = "confidence low" if confidence < 50 else "confidence"
    confidence_note = (
        " — treat this as a rough estimate, the trees disagree a lot on this listing"
        if confidence < 50 else ""
    )

    st.markdown(f"""
    <div class="result">
    <div class="amount">€{point_pred:,.0f} / month</div>
    <div class="{confidence_class}">Confidence: {confidence:.0f}% of trees within ±10% of this estimate{confidence_note}</div>
    <div class="range">80% of individual trees predicted between €{p10:,.0f} and €{p90:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    fig = px.histogram(
        x=tree_preds_flat, nbins=30,
        title=f"{len(tree_preds_flat)} individual tree predictions for this listing",
        labels={"x": "Predicted rent (€) per tree"},
        color_discrete_sequence=["#C1592E"],
    )
    fig.add_vline(x=point_pred, line_color="#4C6570", annotation_text="forest average")
    fig.update_layout(yaxis_title="# trees", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why this estimate? (feature breakdown)"):
        feature_names = get_feature_names(pipeline)
        explainer = build_explainer(pipeline)
        preprocessor = pipeline.named_steps["preprocessor"]
        X_processed = preprocessor.transform(new_data)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        shap_row = explainer(X_processed)
        exp = shap.Explanation(
            values=shap_row.values[0],
            base_values=shap_row.base_values[0],
            data=X_processed[0],
            feature_names=feature_names,
        )
        fig_waterfall = plt.figure()
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig_waterfall, clear_figure=True)
        plt.close(fig_waterfall)
        st.caption(
            "Starts from the average predicted rent across all listings, then shows how "
            "each feature of this specific apartment pushed the estimate up or down."
        )

# Sidebar navigation

st.sidebar.title("Berlin Rent Predictor")
st.sidebar.radio(
    "Sections",
    SECTIONS,
    key="current_page",
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.8rem;opacity:0.75'>
<b>Data source:</b> Berlin listings (Immowelt)<br>
<b>Model:</b> Random Forest<br>
<b>Target:</b> Cold rent (€)
</div>
""", unsafe_allow_html=True)


# Routing

page = st.session_state.current_page

if page == "Overview":
    render_overview()
elif page == "Explore the Data":
    render_explore_page()
elif page == "Model & Performance":
    render_model_page()
elif page == "Model Card":
    render_model_card_page()
elif page == "Predict Your Rent":
    render_predict_page()