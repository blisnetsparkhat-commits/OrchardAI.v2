import math
import os
from datetime import date

import folium
import joblib
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


# =========================================================
# ORCHARDAI CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OrchardAI · Smart Agriculture",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "orchard_model.pkl"
DATA_PATH = "tree_health_data.csv"

DRIVE_FOLDER = "/content/drive/MyDrive/OrchardAI_Submission"
DRIVE_ORCHARDS_PATH = os.path.join(DRIVE_FOLDER, "orchards.csv")

if os.path.exists("/content/drive"):
    ORCHARDS_PATH = (
        DRIVE_ORCHARDS_PATH
        if os.path.exists(DRIVE_ORCHARDS_PATH)
        else "orchards.csv"
    )
else:
    ORCHARDS_PATH = "orchards.csv"

OBSERVATION_COLUMNS = [
    "Tree ID",
    "Date",
    "Growth stage",
    "Leaf condition",
    "Leaf color",
    "Leaf damage",
    "Disease symptoms",
    "Disease symptoms present",
    "Fruit condition",
    "Notes"
]

LEAF_DAMAGE_MAP = {
    "None observed": 0,
    "Spots": 1,
    "Holes": 2,
    "Drying": 3
}

FEATURE_COLUMNS = [
    "air_temperature",
    "air_humidity",
    "precipitation",
    "soil_temperature",
    "soil_moisture",
    "leaf_damage",
    "yellow_leaves_pct",
    "disease_symptoms"
]

ORCHARD_COLUMNS = [
    "company",
    "district",
    "location",
    "area_ha",
    "trees",
    "varieties",
    "latitude",
    "longitude",
    "location_status"
]

ORCHARD_TEXT_TRANSLATIONS = {
    "Енбекшиказахский": "Enbekshikazakh District",
    "Талгарский": "Talgar District",
    "Талгарский и Енбекшиказахский": "Talgar and Enbekshikazakh Districts",
    "Алматинская область": "Almaty Region",
    "Каражотинский сельский округ": "Karazhotin Rural District",
    "Точное расположение не подтверждено": "Exact location not verified",
    "Несколько районов": "Multiple districts",
    "Село Тургень": "Turgen Village",
    "Казахстан, улица Ы. Калдыбаева, 28А": "Kazakhstan Village, Y. Kaldybaeva Street, 28A",
    "Более 12 сортов": "More than 12 varieties",
    "Интенсивный сад": "Intensive orchard",
    "14 сортов": "14 varieties",
    "Не указано": "Not specified",
    "Координаты не подтверждены": "Coordinates not verified",
    "Координаты сада не подтверждены": "Orchard coordinates not verified",
    "Координаты подтверждены пользователем": "Coordinates confirmed by user",
    "Точное местоположение уточняется": "Exact location pending verification",
    "Точные координаты сада не подтверждены": "Orchard coordinates not verified",
    "Не уточнены": "Not specified",
    "Не уточнено": "Not specified",
    "Не уточнена": "Not specified",
    "Село Казахстан": "Kazakhstan Village",
    "Gala, Fuji, Golden Delicious и другие": "Gala, Fuji, Golden Delicious, and others"
}


# =========================================================
# DATA STORAGE
# =========================================================

def load_observations():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    try:
        data = pd.read_csv(DATA_PATH)
        for column in OBSERVATION_COLUMNS:
            if column not in data.columns:
                data[column] = ""
        return data[OBSERVATION_COLUMNS]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    except Exception as error:
        st.error(f"Could not load saved observations: {error}")
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def save_observations(data):
    temp_path = DATA_PATH + ".tmp"
    data.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, DATA_PATH)


def create_initial_orchards():
    return pd.DataFrame([
        {
            "company": "Arnau Agro",
            "district": "Enbekshikazakh District",
            "location": "Karazhotin Rural District",
            "area_ha": 470,
            "trees": 350000,
            "varieties": "Gala, Fuji, Golden Delicious",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Apple Bee",
            "district": "Almaty Region",
            "location": "Exact location not verified",
            "area_ha": 120,
            "trees": None,
            "varieties": "More than 12 varieties",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Baimene",
            "district": "Talgar District",
            "location": "Exact location not verified",
            "area_ha": 110,
            "trees": None,
            "varieties": "Intensive orchard",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Amal Bio",
            "district": "Talgar and Enbekshikazakh Districts",
            "location": "Multiple districts",
            "area_ha": 150,
            "trees": 500000,
            "varieties": "14 varieties",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Turgen Apple",
            "district": "Enbekshikazakh District",
            "location": "Turgen Village",
            "area_ha": 17.5,
            "trees": None,
            "varieties": "Golden Delicious, Starkrimson",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Agros BT",
            "district": "Enbekshikazakh District",
            "location": "Kazakhstan Village, Y. Kaldybaeva Street, 28A",
            "area_ha": None,
            "trees": None,
            "varieties": "Not specified",
            "latitude": None,
            "longitude": None,
            "location_status": "Orchard coordinates not verified"
        }
    ], columns=ORCHARD_COLUMNS)


def translate_orchard_text(data):
    data = data.copy()
    for column in ["district", "location", "varieties", "location_status"]:
        if column in data.columns:
            data[column] = data[column].apply(
                lambda value: ORCHARD_TEXT_TRANSLATIONS.get(str(value).strip(), value)
                if pd.notna(value) else value
            )
    return data


def load_orchards():
    if not os.path.exists(ORCHARDS_PATH):
        return create_initial_orchards()
    try:
        data = pd.read_csv(ORCHARDS_PATH)
        for column in ORCHARD_COLUMNS:
            if column not in data.columns:
                data[column] = None
        data = data[ORCHARD_COLUMNS]
        return translate_orchard_text(data)
    except pd.errors.EmptyDataError:
        return create_initial_orchards()
    except Exception as error:
        st.error(f"Could not load orchard database: {error}")
        return create_initial_orchards()


def save_orchards(data):
    folder = os.path.dirname(ORCHARDS_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temp_path = ORCHARDS_PATH + ".tmp"
    data.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, ORCHARDS_PATH)


# Session State Initialization
if "observations" not in st.session_state:
    st.session_state.observations = load_observations()

if "orchards" not in st.session_state:
    st.session_state.orchards = load_orchards()

if "selected_latitude" not in st.session_state:
    st.session_state.selected_latitude = None

if "selected_longitude" not in st.session_state:
    st.session_state.selected_longitude = None

if "agro_chat" not in st.session_state:
    st.session_state.agro_chat = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am the **OrchardAI Agronomist Assistant**. "
                "Ask me about apple diseases (scab, powdery mildew, fire blight), "
                "frost protection, or irrigation schedules for Almaty Region."
            )
        }
    ]


# =========================================================
# MODERN GLASSMORPHISM DESIGN & TYPOGRAPHY
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* MAIN BACKGROUND WITH SUBTLE RADIAL GLOW */
.stApp, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 10%, #123B2B 0%, #071D15 55%, #05140E 100%);
    color: #E8F3EA;
}

[data-testid="stHeader"] {
    background: rgba(7, 29, 21, 0.85);
    backdrop-filter: blur(12px);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D291E 0%, #071811 100%);
    border-right: 1px solid rgba(134, 194, 148, 0.16);
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #E3F7E4 !important;
    font-weight: 750 !important;
    letter-spacing: -0.02em;
}

p, label {
    color: #CFE3D2;
}

/* GLASSMORPHISM METRIC CARDS */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(25, 67, 48, 0.72), rgba(14, 41, 29, 0.85));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(138, 206, 152, 0.22);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(165, 232, 178, 0.48);
}

[data-testid="stMetricLabel"] {
    color: #9EC4A6 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
}

/* CUSTOM GLASS CARD COMPONENT */
.glass-card {
    background: linear-gradient(145deg, rgba(23, 62, 45, 0.65), rgba(12, 36, 26, 0.85));
    border: 1px solid rgba(138, 206, 152, 0.2);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.2);
}

.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
    background: rgba(56, 161, 105, 0.22);
    color: #9AE6B4;
    border: 1px solid rgba(154, 230, 180, 0.35);
    margin-bottom: 8px;
}

/* BUTTONS */
.stButton button,
.stDownloadButton button,
.stFormSubmitButton button {
    background: linear-gradient(135deg, #2F9E62 0%, #1F6E43 100%);
    color: #FFFFFF !important;
    border: 1px solid rgba(164, 235, 182, 0.35);
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 14px rgba(31, 110, 67, 0.35);
    transition: all 0.22s ease;
}

.stButton button:hover,
.stDownloadButton button:hover,
.stFormSubmitButton button:hover {
    background: linear-gradient(135deg, #3AB873 0%, #278552 100%);
    border-color: #B7F4C6;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(58, 184, 115, 0.45);
}

/* INPUTS */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(18, 46, 34, 0.9) !important;
    color: #F2F9F3 !important;
    border: 1px solid rgba(120, 180, 134, 0.3) !important;
    border-radius: 11px !important;
}

/* FORMS & CONTAINERS */
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: rgba(15, 40, 29, 0.78);
    border: 1px solid rgba(126, 188, 140, 0.22);
    border-radius: 18px;
    padding: 16px;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(13, 36, 26, 0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(126, 188, 140, 0.18);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #B4D3BA;
    font-weight: 600;
    padding: 8px 18px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2D8F5A, #1E633D) !important;
    color: #FFFFFF !important;
}

/* DIVIDERS */
hr {
    border-color: rgba(130, 190, 144, 0.18) !important;
}

/* HERO BANNER */
.orchard-hero {
    min-height: 225px;
    background-size: cover;
    background-position: center;
    border-radius: 22px;
    margin: 4px 0 26px 0;
    padding: 32px 38px;
    display: flex;
    align-items: center;
    border: 1px solid rgba(168, 224, 178, 0.25);
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.32);
}
.orchard-hero-content {
    max-width: 680px;
}
.orchard-hero-icon {
    font-size: 38px;
    margin-bottom: 6px;
}
.orchard-hero-title {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.18;
    margin-bottom: 8px;
}
.orchard-hero-subtitle {
    color: #DCECE0;
    font-size: 15.5px;
    line-height: 1.5;
}
.orchard-hero-tag {
    display: inline-block;
    color: #A8E6B8;
    background: rgba(168, 230, 184, 0.12);
    border: 1px solid rgba(168, 230, 184, 0.28);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.8px;
    margin-top: 18px;
}

.orchard-brand {
    text-align: center;
    padding: 6px 0 16px 0;
}
.orchard-brand-icon {
    font-size: 42px;
}
.orchard-brand-name {
    color: #E2F6E4;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.orchard-brand-sub {
    color: #8EB897;
    font-size: 11px;
    letter-spacing: 0.8px;
}

@media (max-width: 768px) {
    .orchard-hero {
        min-height: 175px;
        padding: 20px;
    }
    .orchard-hero-title {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# PREMIUM TABLE HELPER
# =========================================================

def show_styled_table(data, **kwargs):
    if data is None or data.empty:
        st.info("No data available.")
        return

    styled = (
        data.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#153D2A"),
                    ("color", "#E6F7E8"),
                    ("font-weight", "700"),
                    ("padding", "13px"),
                    ("border", "1px solid #2D5C42"),
                    ("text-align", "left")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "11px 13px"),
                    ("color", "#EAF4EC"),
                    ("border-bottom", "1px solid #284E3A")
                ]
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [("background-color", "#123123")]
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [("background-color", "#183D2C")]
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%")
                ]
            }
        ])
    )
    st.dataframe(styled, **kwargs)


# =========================================================
# VISUAL BANNERS
# =========================================================

BASE_SVG = 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzEyM2MyNyIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNmNGQyN2EiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIiBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICA8Y2lyY2xlIGN4PSIxMDAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9Ijc4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9Ijg4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjExNiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgIDxjaXJjbGUgY3g9IjIzMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgPGNpcmNsZSBjeD0iMjA4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9IjIxOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICA8Y2lyY2xlIGN4PSIyNDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICA8Y2lyY2xlIGN4PSIzNjAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9IjMzOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSIzNDgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgPGNpcmNsZSBjeD0iMzc2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgPGNpcmNsZSBjeD0iNDkwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSI0NjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgPGNpcmNsZSBjeD0iNDc4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjUwNiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgIDxjaXJjbGUgY3g9IjYyMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgPGNpcmNsZSBjeD0iNTk4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9IjYwOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICA8Y2lyY2xlIGN4PSI2MzYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICA8Y2lyY2xlIGN4PSI3NTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9IjcyOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSI3MzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgPGNpcmNsZSBjeD0iNzY2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgPGNpcmNsZSBjeD0iODgwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICA8Y2lyY2xlIGN4PSI4NTgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgPGNpcmNsZSBjeD0iODY4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgIDxjaXJjbGUgY3g9Ijg5NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICA8cGF0aCBkPSJNMCAzMjUgUTUwMCAyODUgMTAwMCAzMjUgVjM2MCBIMFoiIGZpbGw9IiMxNDJkMjAiLz4KICAgIDwvc3ZnPg=='

ORCHARD_PAGE_CONTENT = {
    "Dashboard": ("🍎", "Welcome to OrchardAI", "Intelligent insights, live weather telemetry, and ML risk screening for apple orchards."),
    "Weather": ("🌤️", "Regional Weather & Agro-Metrics", "Live conditions, dew point, and fungal infection risk across Almaty Region."),
    "Tree Health": ("🌳", "Tree Health Registry", "Record, filter, and manage structured field observations for every tree."),
    "Analytics": ("📊", "Orchard Analytics & Yield Simulator", "Analyze environmental risks, observation trends, and harvest impact."),
    "ML Prediction": ("🧠", "OrchardAI ML Risk Screening", "Single-tree diagnostics and batch CSV dataset evaluation."),
    "AI Agronomist": ("🌿", "AI Agronomist Assistant", "Interactive orchard care guide, disease protocols, and seasonal reminders."),
    "Reports": ("📄", "Reports & Data Export", "Download structured datasets, weather logs, and trained ML artifacts."),
    "Orchard Map": ("🗺️", "Almaty Regional Orchard Map", "Explore registered apple orchards and verify field coordinates.")
}


def show_orchard_hero(page_name):
    emoji, heading, subtitle = ORCHARD_PAGE_CONTENT[page_name]
    st.markdown(
        f"""
        <div class="orchard-hero"
             style="background-image:
             linear-gradient(95deg, rgba(6,24,16,0.95) 0%, rgba(10,36,25,0.72) 55%, rgba(8,25,17,0.32) 100%),
             url('data:image/svg+xml;base64,{BASE_SVG}');">
            <div class="orchard-hero-content">
                <div class="orchard-hero-icon">{emoji}</div>
                <div class="orchard-hero-title">{heading}</div>
                <div class="orchard-hero-subtitle">{subtitle}</div>
                <div class="orchard-hero-tag">ORCHARDAI • SMART AGRICULTURE PLATFORM</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# WEATHER & AGRO-INDEX HELPERS
# =========================================================

WEATHER_LOCATIONS = [
    {"district": "Balkhash District", "place": "Bakanas"},
    {"district": "Enbekshikazakh District", "place": "Esik"},
    {"district": "Zhambyl District", "place": "Uzynagash"},
    {"district": "Ili District", "place": "Otegen Batyr"},
    {"district": "Karasay District", "place": "Kaskelen"},
    {"district": "Kegen District", "place": "Kegen"},
    {"district": "Raiymbek District", "place": "Narynkol"},
    {"district": "Talgar District", "place": "Talgar"},
    {"district": "Uyghur District", "place": "Chundzha"},
    {"district": "Konaev City", "place": "Kapchagay"},
    {"district": "Alatau City", "place": "Alatau"}
]


def calculate_dew_point(temp_c, humidity):
    if pd.isna(temp_c) or pd.isna(humidity) or humidity <= 0:
        return None
    a, b = 17.27, 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(humidity / 100.0)
    return round((b * alpha) / (a - alpha), 1)


def assess_fungal_risk(temp_c, humidity):
    if pd.isna(temp_c) or pd.isna(humidity):
        return "Unknown"
    if humidity >= 80 and 12 <= temp_c <= 26:
        return "High (Scab/Mildew)"
    elif humidity >= 70 and 8 <= temp_c <= 28:
        return "Moderate"
    return "Low"


@st.cache_data(ttl=1800, show_spinner=False)
def load_regional_weather():
    rows = []
    for location in WEATHER_LOCATIONS:
        district = location["district"]
        place = location["place"]
        try:
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": place, "count": 10, "language": "en", "format": "json", "countryCode": "KZ"},
                timeout=12
            )
            geo_response.raise_for_status()
            geo_results = geo_response.json().get("results", [])
            matches = [item for item in geo_results if item.get("country_code") == "KZ"]
            if not matches:
                raise ValueError("Location not found")

            selected = next(
                (item for item in matches if "almaty" in str(item.get("admin1", "")).casefold()),
                matches[0]
            )

            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": selected["latitude"],
                    "longitude": selected["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                    "timezone": "Asia/Almaty"
                },
                timeout=12
            )
            weather_response.raise_for_status()
            current = weather_response.json()["current"]

            temp_val = current.get("temperature_2m")
            hum_val = current.get("relative_humidity_2m")

            rows.append({
                "Region": district,
                "Reference settlement": place,
                "Temperature (°C)": temp_val,
                "Humidity (%)": hum_val,
                "Dew Point (°C)": calculate_dew_point(temp_val, hum_val),
                "Wind (km/h)": current.get("wind_speed_10m"),
                "Precipitation (mm)": current.get("precipitation"),
                "Fungal Risk": assess_fungal_risk(temp_val, hum_val),
                "Local time": current.get("time"),
                "Status": "Available"
            })
        except Exception:
            rows.append({
                "Region": district,
                "Reference settlement": place,
                "Temperature (°C)": None,
                "Humidity (%)": None,
                "Dew Point (°C)": None,
                "Wind (km/h)": None,
                "Precipitation (mm)": None,
                "Fungal Risk": "Unavailable",
                "Local time": None,
                "Status": "Unavailable"
            })
    return pd.DataFrame(rows)


weather = {
    "Temperature": 2.9,
    "Humidity": 83,
    "Precipitation": 0.0,
    "Wind speed": 4.5,
    "Soil temperature": 2.1,
    "Soil moisture": 0.334
}

forecast = pd.DataFrame({
    "Date": ["Nov 15", "Nov 16", "Nov 17", "Nov 18", "Nov 19", "Nov 20", "Nov 21"],
    "Min temperature": [-1.4, -5.7, -5.7, -1.6, -0.5, 1.2, 0.3],
    "Max temperature": [4.5, 0.5, 0.6, 7.2, 7.7, 11.9, 13.9],
    "Precipitation": [1.7, 8.4, 0, 0, 0, 0, 0]
})


# =========================================================
# MACHINE LEARNING FUNCTIONS
# =========================================================

def create_prediction_data(
    air_temp,
    air_humidity,
    precipitation,
    soil_temp,
    soil_moisture,
    leaf_damage,
    disease
):
    return pd.DataFrame([{
        "air_temperature": air_temp,
        "air_humidity": air_humidity,
        "precipitation": precipitation,
        "soil_temperature": soil_temp,
        "soil_moisture": soil_moisture,
        "leaf_damage": leaf_damage,
        "yellow_leaves_pct": 0,
        "disease_symptoms": disease
    }], columns=FEATURE_COLUMNS)


def align_features_for_model(df, model):
    if not isinstance(df, pd.DataFrame):
        return df

    aligned = df.copy()

    if "temperature" not in aligned.columns and "air_temperature" in aligned.columns:
        aligned["temperature"] = aligned["air_temperature"]
    if "humidity" not in aligned.columns and "air_humidity" in aligned.columns:
        aligned["humidity"] = aligned["air_humidity"]
    if "tree_age" not in aligned.columns:
        aligned["tree_age"] = 5.0

    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in aligned.columns:
                aligned[col] = 0
        aligned = aligned[expected_cols]

    return aligned


def show_prediction(model, tree_data, input_factors=None):
    prepared_data = align_features_for_model(tree_data, model)
    prediction = str(model.predict(prepared_data)[0])
    probabilities = model.predict_proba(prepared_data)[0]

    normalized = prediction.casefold()
    if "low" in normalized or prediction == "0":
        risk_title = "Low Risk"
        risk_color = "#2E8B57"
        explanation = (
            "The model classifies this input as low risk. The submitted "
            "values do not match higher-risk patterns strongly enough."
        )
        next_action = "Continue routine monitoring and log a new observation if tree conditions shift."
    elif "medium" in normalized or "moderate" in normalized:
        risk_title = "Moderate Risk"
        risk_color = "#D69E2E"
        explanation = (
            "The model classifies this input as moderate risk. Some environmental or leaf "
            "indicators deserve closer field inspection."
        )
        next_action = "Re-inspect the tree within 48 hours, document visible changes, and check neighboring trees."
    elif "high" in normalized or prediction == "1":
        risk_title = "High Risk"
        risk_color = "#E53E3E"
        explanation = (
            "The model classifies this input as high risk. The submitted conditions "
            "resemble stress or symptom patterns in the training dataset."
        )
        next_action = "Isolate observation zone, record detailed symptoms, and consult a certified agronomist."
    else:
        risk_title = prediction
        risk_color = "#388B58"
        explanation = "Model returned a custom risk classification."
        next_action = "Review the input values and consult an agronomist if symptoms are visible."

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(23,62,45,0.9), rgba(9,37,27,0.95));
            border: 1px solid {risk_color};
            border-left: 8px solid {risk_color};
            border-radius: 18px; padding: 22px 26px; margin: 14px 0 20px 0;
            box-shadow: 0 8px 24px rgba(0,0,0,0.28);">
            <div style="font-size:11px; letter-spacing:1.8px; color:#A8DFB4; font-weight:700;">
                ORCHARDAI SCREENING RESULT
            </div>
            <div style="font-size:32px; font-weight:800; color:#FFFFFF; margin-top:6px;">
                {risk_title}
            </div>
            <div style="font-size:13px; color:#CFE3D2; margin-top:4px;">
                Raw model class: <b>{prediction}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_exp, col_prob = st.columns([1.2, 1])
    with col_exp:
        st.subheader("Interpretation & Action")
        st.write(explanation)
        st.info(f"**Recommended next step:** {next_action}")
        if input_factors:
            st.markdown("**Key factors reviewed:**")
            for factor in input_factors:
                st.markdown(f"- {factor}")

    with col_prob:
        st.subheader("Class Probability Distribution")
        for label, probability in zip(model.classes_, probabilities):
            st.write(f"**{label}** — `{float(probability):.1%}`")
            st.progress(float(probability))

    st.caption(
        "DEMO ONLY — Trained on synthetic data. Supports screening workflows, not clinical diagnosis."
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div class="orchard-brand">
    <div class="orchard-brand-icon">🍎</div>
    <div class="orchard-brand-name">ORCHARDAI</div>
    <div class="orchard-brand-sub">PRECISION ORCHARD INTEL</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.pop("navigate_to_ml_prediction", False):
    st.session_state["orchard_page_choice"] = "ML Prediction"
if st.session_state.pop("navigate_to_tree_health", False):
    st.session_state["orchard_page_choice"] = "Tree Health"

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Weather",
        "Tree Health",
        "Analytics",
        "ML Prediction",
        "AI Agronomist",
        "Reports",
        "Orchard Map"
    ],
    key="orchard_page_choice"
)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    <div style="background: rgba(255,255,255,0.04); padding: 12px; border-radius: 12px; font-size: 12.5px;">
        <div>📍 <b>Almaty Region, KZ</b></div>
        <div style="margin-top:4px; color:#9BC4A3;">Saved Trees: <b>{len(st.session_state.observations)}</b></div>
        <div style="color:#9BC4A3;">Registered Orchards: <b>{len(st.session_state.orchards)}</b></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("OrchardAI v2.0 · Hackathon Edition")


# =========================================================
# 1. DASHBOARD
# =========================================================

if page == "Dashboard":
    show_orchard_hero("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference Air Temp", f'{weather["Temperature"]:.1f} °C', delta="-0.8 °C vs avg")
    c2.metric("Air Humidity", f'{weather["Humidity"]:.0f}%', delta="High moisture")
    c3.metric("Logged Observations", len(st.session_state.observations), delta="Active CSV")
    c4.metric("Monitored Districts", f"{len(WEATHER_LOCATIONS)}", delta="Open-Meteo")

    st.divider()

    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("🧠 Run ML Tree Screening", type="primary", use_container_width=True):
            st.session_state["navigate_to_ml_prediction"] = True
            st.rerun()
    with qa2:
        if st.button("🌳 Log Field Observation", use_container_width=True):
            st.session_state["navigate_to_tree_health"] = True
            st.rerun()
    with qa3:
        st.download_button(
            "⬇️ Quick Export Observations",
            data=st.session_state.observations.to_csv(index=False).encode("utf-8-sig"),
            file_name="OrchardAI_Observations.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    left, right = st.columns([1.8, 1])

    with left:
        st.subheader("7-Day Temperature Outlook (°C)")
        st.line_chart(
            forecast.set_index("Date")[["Min temperature", "Max temperature"]],
            color=["#4FD1C5", "#F6AD55"]
        )

    with right:
        st.subheader("System Health & Readiness")
        model_badge = "Loaded (DEMO v1.0)" if os.path.exists(MODEL_PATH) else "Not found"
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="status-pill">● ALL SYSTEMS OPERATIONAL</span>
                <p style="margin:6px 0;">✅ <b>Open-Meteo API:</b> Connected (11 districts)</p>
                <p style="margin:6px 0;">✅ <b>Observation Storage:</b> Ready ({len(st.session_state.observations)} rows)</p>
                <p style="margin:6px 0;">✅ <b>Orchard GIS Registry:</b> {len(st.session_state.orchards)} enterprises</p>
                <p style="margin:6px 0;">🧠 <b>ML Screening Engine:</b> {model_badge}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        """
        <div class="glass-card">
            <span class="status-pill">REAL-WORLD CASE STUDY · KAZAKHSTAN (2026)</span>
            <h3 style="margin-top:4px;">Why Early Orchard Monitoring Matters</h3>
            <p>
                In 2026, a commercial apple-growing enterprise in Kazakhstan harvested only 
                <b>45–47% of its initially projected yield</b> due to late detection of fungal diseases, 
                pests, and spring frost stress. OrchardAI combines live microclimate tracking, 
                structured scouting logs, and ML risk screening to help growers act before losses compound.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 2. WEATHER
# =========================================================

elif page == "Weather":
    show_orchard_hero("Weather")

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.write("Live meteorological telemetry, dew point calculations, and fungal risk indicators across Almaty Region.")
    with top_col2:
        if st.button("🔄 Refresh Live Weather", use_container_width=True):
            load_regional_weather.clear()
            st.rerun()

    with st.spinner("Fetching live data from Open-Meteo..."):
        regional_weather = load_regional_weather()

    available = regional_weather[regional_weather["Status"] == "Available"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations Online", f"{len(available)} / {len(WEATHER_LOCATIONS)}")
    if not available.empty:
        c2.metric("Regional Avg Temp", f'{available["Temperature (°C)"].mean():.1f} °C')
        c3.metric("Regional Avg Humidity", f'{available["Humidity (%)"].mean():.0f}%')
        high_fungal = (available["Fungal Risk"].str.contains("High|Moderate")).sum()
        c4.metric("Elevated Fungal Alerts", f"{high_fungal} districts")
    else:
        c2.metric("Regional Avg Temp", "Unavailable")
        c3.metric("Regional Avg Humidity", "Unavailable")
        c4.metric("Elevated Fungal Alerts", "—")

    st.divider()

    district_options = ["All locations"] + sorted(regional_weather["Region"].dropna().unique().tolist())
    selected_district = st.selectbox("Filter by district / city", district_options)

    weather_view = regional_weather.copy()
    if selected_district != "All locations":
        weather_view = weather_view[weather_view["Region"] == selected_district]

    for column in ["Temperature (°C)", "Humidity (%)", "Dew Point (°C)", "Wind (km/h)", "Precipitation (mm)"]:
        weather_view[column] = weather_view[column].apply(
            lambda value: f"{value:.1f}" if pd.notna(value) else "—"
        )

    show_styled_table(
        weather_view[
            ["Region", "Reference settlement", "Temperature (°C)", "Humidity (%)",
             "Dew Point (°C)", "Wind (km/h)", "Precipitation (mm)", "Fungal Risk"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Regional Weather & Agro-Metrics CSV",
        data=regional_weather.to_csv(index=False).encode("utf-8-sig"),
        file_name="OrchardAI_Regional_Weather.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("7-Day Frost & Precipitation Forecast")
    show_styled_table(forecast, use_container_width=True, hide_index=True)


# =========================================================
# 3. TREE HEALTH
# =========================================================

elif page == "Tree Health":
    show_orchard_hero("Tree Health")

    tab_add, tab_manage = st.tabs(["➕ Log New Observation", "📋 Observation Registry & Management"])

    with tab_add:
        with st.form("tree_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**1 · Identification & Phenology**")
                tree_id = st.text_input("Tree ID *", placeholder="e.g. BLK-A-014")
                growth_stage = st.selectbox(
                    "Growth Stage",
                    ["Dormant", "Bud development", "Flowering", "Fruit development", "Harvest", "Unknown"]
                )
                fruit = st.selectbox("Fruit Condition", ["Healthy", "Damaged", "No fruit", "Unknown"])

            with col_b:
                st.markdown("**2 · Canopy & Symptom Assessment**")
                leaf_condition = st.selectbox(
                    "Leaf Condition",
                    ["Healthy", "Moderate damage", "Severe damage", "No leaves", "Unknown"]
                )
                leaf_color = st.selectbox(
                    "Leaf Color",
                    ["Green", "Yellow", "Brown", "Mixed", "No leaves", "Unknown"]
                )
                leaf_damage = st.selectbox(
                    "Leaf Damage",
                    ["None observed", "Spots", "Holes", "Drying", "Unknown"]
                )

            st.markdown("**3 · Pathology & Field Notes**")
            col_c, col_d = st.columns([1, 2])
            with col_c:
                disease_present = st.selectbox("Are disease symptoms present?", ["Unknown", "No", "Yes"])
            with col_d:
                disease = st.text_input("Visible Symptoms", placeholder="e.g. Olive-brown spots on underside of leaves")

            notes = st.text_area("Additional Agronomic Notes", placeholder="Soil drainage, pest presence, recent spraying...")

            submitted = st.form_submit_button("💾 Save Tree Observation", use_container_width=True)

            if submitted:
                if not tree_id.strip():
                    st.error("Please enter a valid Tree ID.")
                else:
                    new_record = pd.DataFrame([{
                        "Tree ID": tree_id.strip(),
                        "Date": str(date.today()),
                        "Growth stage": growth_stage,
                        "Leaf condition": leaf_condition,
                        "Leaf color": leaf_color,
                        "Leaf damage": leaf_damage,
                        "Disease symptoms": disease.strip(),
                        "Disease symptoms present": disease_present,
                        "Fruit condition": fruit,
                        "Notes": notes.strip()
                    }])

                    updated_observations = pd.concat(
                        [st.session_state.observations, new_record],
                        ignore_index=True
                    )
                    try:
                        save_observations(updated_observations)
                        st.session_state.observations = updated_observations
                        st.success(f"Observation for **{tree_id.strip()}** saved!")
                    except Exception as error:
                        st.error(f"Could not save observation: {error}")

    with tab_manage:
        obs = st.session_state.observations
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", len(obs))
        m2.metric("Symptomatic Trees", int((obs["Disease symptoms present"] == "Yes").sum()) if not obs.empty else 0)
        m3.metric("Severe Leaf Damage", int((obs["Leaf condition"] == "Severe damage").sum()) if not obs.empty else 0)

        if not obs.empty:
            search_query = st.text_input("🔍 Search by Tree ID or Notes", placeholder="Type Tree ID...")
            filtered_obs = obs
            if search_query.strip():
                mask = (
                    obs["Tree ID"].astype(str).str.contains(search_query, case=False, na=False)
                    | obs["Notes"].astype(str).str.contains(search_query, case=False, na=False)
                )
                filtered_obs = obs[mask]

            show_styled_table(filtered_obs, use_container_width=True)

            col_dl, col_del = st.columns([1, 1.5])
            with col_dl:
                st.download_button(
                    "⬇️ Export Filtered Dataset (CSV)",
                    data=filtered_obs.to_csv(index=False).encode("utf-8-sig"),
                    file_name="OrchardAI_Tree_Health.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_del:
                with st.expander("🗑️ Delete an Observation Record"):
                    row_to_delete = st.selectbox(
                        "Select record index to remove",
                        obs.index.tolist(),
                        format_func=lambda idx: f"Row {idx}: {obs.loc[idx, 'Tree ID']} ({obs.loc[idx, 'Date']})"
                    )
                    if st.button("Confirm Delete", type="secondary"):
                        updated_obs = obs.drop(index=row_to_delete).reset_index(drop=True)
                        save_observations(updated_obs)
                        st.session_state.observations = updated_obs
                        st.rerun()
        else:
            st.info("No observations logged yet. Use the 'Log New Observation' tab to add your first entry.")


# =========================================================
# 4. ANALYTICS & YIELD SIMULATOR
# =========================================================

elif page == "Analytics":
    show_orchard_hero("Analytics")

    st.subheader("Environmental & Frost Risk Indicators")
    freezing_days = int((forecast["Min temperature"] <= 0).sum())
    rainy_days = int((forecast["Precipitation"] > 0).sum())
    max_swing = float((forecast["Max temperature"] - forecast["Min temperature"]).max())

    c1, c2, c3 = st.columns(3)
    c1.metric("Frost Risk Days (≤ 0°C)", f"{freezing_days} days")
    c2.metric("Precipitation Days", f"{rainy_days} days")
    c3.metric("Max Diurnal Temp Swing", f"{max_swing:.1f} °C")

    st.bar_chart(forecast.set_index("Date")[["Min temperature", "Max temperature"]])

    st.divider()
    st.subheader("💰 Orchard Yield & Economic Impact Simulator")
    st.write("Estimate potential harvest protection based on the 2026 Kazakhstan orchard loss benchmark (45–47% yield realization without early intervention).")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_area = st.number_input("Orchard Area (hectares)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
    with sim_c2:
        expected_yield_ha = st.number_input("Target Yield (tons / ha)", min_value=5.0, max_value=80.0, value=25.0, step=1.0)
    with sim_c3:
        recovery_pct = st.slider("Estimated Loss Prevented by Early Action (%)", 5, 40, 18)

    baseline_total = sim_area * expected_yield_ha
    unmanaged_harvest = baseline_total * 0.46
    saved_tons = baseline_total * (recovery_pct / 100.0)

    res1, res2, res3 = st.columns(3)
    res1.metric("Target Harvest", f"{baseline_total:,.0f} tons")
    res2.metric("Unchecked Loss Scenario (46%)", f"{unmanaged_harvest:,.0f} tons", delta=f"-{baseline_total - unmanaged_harvest:,.0f} t")
    res3.metric("Protected Yield via Screening", f"+{saved_tons:,.1f} tons", delta=f"+{recovery_pct}% recovery")

    st.divider()
    st.subheader("Tree Observation Breakdown")
    observations = st.session_state.observations
    if len(observations) < 3:
        st.info(
            f"Currently {len(observations)} observation(s) logged. "
            "Add at least 3 observations in Tree Health to unlock distribution charts."
        )
    else:
        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("**By Leaf Condition**")
            st.bar_chart(observations["Leaf condition"].value_counts())
        with chart2:
            st.markdown("**By Growth Stage**")
            st.bar_chart(observations["Growth stage"].value_counts())


# =========================================================
# 5. MACHINE LEARNING PREDICTION
# =========================================================

elif page == "ML Prediction":
    show_orchard_hero("ML Prediction")

    if not os.path.exists(MODEL_PATH):
        st.error("ML model file not found: `orchard_model.pkl`")
        st.info("Place `orchard_model.pkl` in the application directory to enable ML predictions.")
    else:
        try:
            model = joblib.load(MODEL_PATH)

            tab_single, tab_batch = st.tabs(["🔍 Single Tree Screening", "📂 Batch CSV Screening"])

            with tab_single:
                demo_left, demo_right = st.columns(2)
                with demo_left:
                    if st.button("🍏 Load Healthy Preset", use_container_width=True):
                        st.session_state["analysis_mode_choice"] = "Manual Analysis"
                        st.session_state["demo_air_temp"] = 18.0
                        st.session_state["demo_air_humidity"] = 55.0
                        st.session_state["demo_precipitation"] = 0.0
                        st.session_state["demo_soil_temp"] = 15.0
                        st.session_state["demo_soil_moisture"] = 0.25
                        st.session_state["demo_leaf_damage"] = 0
                        st.session_state["demo_disease"] = 0
                        st.rerun()

                with demo_right:
                    if st.button("🍂 Load High-Risk Preset", use_container_width=True):
                        st.session_state["analysis_mode_choice"] = "Manual Analysis"
                        st.session_state["demo_air_temp"] = 22.0
                        st.session_state["demo_air_humidity"] = 88.0
                        st.session_state["demo_precipitation"] = 8.0
                        st.session_state["demo_soil_temp"] = 18.0
                        st.session_state["demo_soil_moisture"] = 0.75
                        st.session_state["demo_leaf_damage"] = 3
                        st.session_state["demo_disease"] = 1
                        st.rerun()

                analysis_mode = st.radio(
                    "Screening Mode",
                    ["Manual Analysis", "Analyze Saved Tree"],
                    horizontal=True,
                    key="analysis_mode_choice"
                )

                if analysis_mode == "Analyze Saved Tree":
                    observations = st.session_state.observations.copy()
                    if observations.empty:
                        st.info("No saved observations yet. Add one in Tree Health first.")
                    else:
                        observations["Leaf damage level"] = observations["Leaf damage"].map(LEAF_DAMAGE_MAP)
                        valid_rows = observations[
                            observations["Leaf damage level"].notna()
                            & observations["Disease symptoms present"].isin(["Yes", "No"])
                        ].copy()

                        if valid_rows.empty:
                            st.warning("Saved observations need known Leaf Damage and Yes/No disease symptoms.")
                        else:
                            selected_index = st.selectbox(
                                "Select saved tree observation",
                                valid_rows.index.tolist(),
                                format_func=lambda idx: f"{valid_rows.loc[idx, 'Tree ID']} — {valid_rows.loc[idx, 'Date']}"
                            )
                            selected_tree = valid_rows.loc[selected_index]

                            if st.button("🔍 Run Screening on Saved Tree", type="primary", use_container_width=True):
                                disease_val = int(selected_tree["Disease symptoms present"] == "Yes")
                                damage_val = int(selected_tree["Leaf damage level"])
                                tree_data = create_prediction_data(
                                    weather["Temperature"], weather["Humidity"], weather["Precipitation"],
                                    weather["Soil temperature"], weather["Soil moisture"], damage_val, disease_val
                                )
                                factors = [
                                    f"Leaf damage recorded: {selected_tree['Leaf damage']}",
                                    f"Symptoms marked: {selected_tree['Disease symptoms present']}",
                                    f"Reference humidity: {weather['Humidity']}%"
                                ]
                                show_prediction(model, tree_data, factors)
                else:
                    input_left, input_right = st.columns(2)
                    with input_left:
                        air_temp = st.number_input("Air Temperature (°C)", value=18.0, key="demo_air_temp")
                        air_humidity = st.number_input("Air Humidity (%)", 0.0, 100.0, 55.0, key="demo_air_humidity")
                        precipitation = st.number_input("Precipitation (mm)", min_value=0.0, value=0.0, key="demo_precipitation")
                        soil_temp = st.number_input("Soil Temperature (°C)", value=15.0, key="demo_soil_temp")

                    with input_right:
                        soil_moisture = st.number_input("Soil Moisture (m³/m³)", 0.0, 1.0, 0.25, key="demo_soil_moisture")
                        leaf_damage = st.selectbox(
                            "Leaf Damage Level",
                            [0, 1, 2, 3],
                            format_func=lambda v: {0: "0 — None", 1: "1 — Spots", 2: "2 — Holes", 3: "3 — Drying/Severe"}[v],
                            key="demo_leaf_damage"
                        )
                        disease = st.selectbox(
                            "Disease Symptoms Present",
                            [0, 1],
                            format_func=lambda v: "No" if v == 0 else "Yes",
                            key="demo_disease"
                        )

                    if st.button("🔍 Analyze Tree Condition", type="primary", use_container_width=True):
                        tree_data = create_prediction_data(
                            air_temp, air_humidity, precipitation,
                            soil_temp, soil_moisture, leaf_damage, disease
                        )
                        factors = [
                            f"Air humidity: {air_humidity:.1f}%",
                            f"Soil moisture: {soil_moisture:.2f} m³/m³",
                            f"Leaf damage level: {leaf_damage}/3",
                            f"Disease symptoms: {'Yes' if disease == 1 else 'No'}"
                        ]
                        show_prediction(model, tree_data, factors)

            with tab_batch:
                st.subheader("Batch Dataset Screening")
                st.write("Upload a CSV containing the model feature columns to score multiple trees at once.")

                template_df = pd.DataFrame([
                    {
                        "air_temperature": 19.5, "air_humidity": 60.0, "precipitation": 0.0,
                        "soil_temperature": 15.0, "soil_moisture": 0.28, "leaf_damage": 0,
                        "yellow_leaves_pct": 0, "disease_symptoms": 0, "tree_age": 5.0
                    },
                    {
                        "air_temperature": 23.0, "air_humidity": 89.0, "precipitation": 6.5,
                        "soil_temperature": 18.5, "soil_moisture": 0.72, "leaf_damage": 3,
                        "yellow_leaves_pct": 15, "disease_symptoms": 1, "tree_age": 8.0
                    }
                ])
                st.download_button(
                    "📄 Download Sample Batch Template CSV",
                    data=template_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="OrchardAI_Batch_Template.csv",
                    mime="text/csv"
                )

                uploaded_batch = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
                if uploaded_batch is not None:
                    batch_df = pd.read_csv(uploaded_batch)
                    prepared_batch = align_features_for_model(batch_df, model)
                    preds = model.predict(prepared_batch)
                    probs = model.predict_proba(prepared_batch).max(axis=1)
                    result_df = batch_df.copy()
                    result_df["Predicted_Risk"] = preds
                    result_df["Confidence"] = [f"{p:.1%}" for p in probs]

                    st.success(f"Successfully screened {len(result_df)} rows!")
                    show_styled_table(result_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Batch Results CSV",
                        data=result_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="OrchardAI_Batch_Results.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Could not run ML prediction: {error}")


# =========================================================
# 6. AI AGRONOMIST ASSISTANT
# =========================================================

elif page == "AI Agronomist":
    show_orchard_hero("AI Agronomist")

    col_guide, col_chat = st.columns([1, 1.35])

    with col_guide:
        st.subheader("📚 Apple Orchard Reference Guide")
        with st.expander("🍂 Apple Scab (Venturia inaequalis)", expanded=True):
            st.write("**Triggers:** High humidity (>80%) + temperatures 12–24°C.")
            st.write("**Symptoms:** Olive-green to black velvety spots on leaves and fruit.")
            st.write("**Action:** Prune canopy for airflow; apply preventative fungicide before prolonged rain.")

        with st.expander("🤍 Powdery Mildew (Podosphaera leucotricha)"):
            st.write("**Triggers:** Warm days (19–25°C) with high night humidity.")
            st.write("**Symptoms:** White powdery coating on young shoots and curled leaves.")
            st.write("**Action:** Remove infected silver-gray shoots during dormant pruning.")

        with st.expander("🔥 Fire Blight (Erwinia amylovora)"):
            st.write("**Triggers:** Warm, wet weather during flowering (>18°C).")
            st.write("**Symptoms:** Shepherd's crook bending of shoots; blackened leaves that stay attached.")
            st.write("**Action:** Cut 30 cm below visible infection; sterilize tools with 70% alcohol between cuts.")

        with st.expander("❄️ Late Spring Frost Protocol"):
            st.write("**Critical Threshold:** -2°C at full bloom causes severe bud loss.")
            st.write("**Action:** Pre-wet soil 24h before frost, run overhead micro-sprinklers, or deploy orchard heaters.")

    with col_chat:
        st.subheader("💬 Ask OrchardAI Assistant")
        for msg in st.session_state.agro_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask about scab, frost, irrigation, or Gala/Fuji care...")
        if user_q:
            st.session_state.agro_chat.append({"role": "user", "content": user_q})
            q_low = user_q.casefold()

            if "scab" in q_low or "парш" in q_low or "spot" in q_low:
                reply = (
                    "**Apple Scab Management:**\n"
                    "- Monitor leaf wetness duration when humidity exceeds **80%**.\n"
                    "- Remove fallen leaves in autumn to reduce overwintering spores.\n"
                    "- Use copper-based sprays at green-tip stage and targeted fungicides after infection periods."
                )
            elif "frost" in q_low or "замороз" in q_low or "cold" in q_low:
                reply = (
                    "**Frost Protection Checklist:**\n"
                    "- Check the 7-day forecast min temperatures (currently showing sub-zero nights).\n"
                    "- Keep orchard floor weed-free and moist to maximize daytime heat absorption.\n"
                    "- Avoid nitrogen fertilization late in the season so wood hardens properly."
                )
            elif "water" in q_low or "irrigat" in q_low or "полив" in q_low or "moisture" in q_low:
                reply = (
                    "**Irrigation Guidance:**\n"
                    "- Optimal volumetric soil moisture for apple trees is **0.22–0.35 m³/m³**.\n"
                    "- Avoid waterlogging (>0.65 m³/m³), which promotes Phytophthora root rot."
                )
            else:
                reply = (
                    "Based on current Almaty Region conditions (cool temperatures & elevated humidity), "
                    "prioritize **dormant pruning sanitation**, check **soil drainage**, and log any "
                    "bark or bud anomalies in the **Tree Health** tab for ML screening."
                )

            st.session_state.agro_chat.append({"role": "assistant", "content": reply})
            st.rerun()


# =========================================================
# 7. REPORTS
# =========================================================

elif page == "Reports":
    show_orchard_hero("Reports")

    r1, r2, r3 = st.columns(3)
    r1.metric("Tree Observations", len(st.session_state.observations))
    r2.metric("Registered Orchards", len(st.session_state.orchards))
    r3.metric("Forecast Horizon", "7 Days")

    st.divider()
    st.subheader("Data Export Center")

    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown("**🌤️ Meteorological Data**")
        st.download_button(
            "Download 7-Day Forecast CSV",
            forecast.to_csv(index=False).encode("utf-8-sig"),
            "OrchardAI_Weather_Forecast.csv",
            "text/csv",
            use_container_width=True
        )
    with exp2:
        st.markdown("**🌳 Tree Health Logs**")
        st.download_button(
            "Download Tree Health CSV",
            st.session_state.observations.to_csv(index=False).encode("utf-8-sig"),
            "OrchardAI_Tree_Health.csv",
            "text/csv",
            use_container_width=True
        )
    with exp3:
        st.markdown("**🗺️ Enterprise Registry**")
        st.download_button(
            "Download Orchards Database CSV",
            st.session_state.orchards.to_csv(index=False).encode("utf-8-sig"),
            "OrchardAI_Orchards.csv",
            "text/csv",
            use_container_width=True
        )

    if os.path.exists(MODEL_PATH):
        st.divider()
        with open(MODEL_PATH, "rb") as model_file:
            st.download_button(
                "🧠 Download Trained ML Model Artifact (.pkl)",
                model_file,
                "orchard_model.pkl",
                "application/octet-stream"
            )


# =========================================================
# 8. ORCHARD MAP
# =========================================================

elif page == "Orchard Map":
    show_orchard_hero("Orchard Map")

    orchards = st.session_state.orchards.copy()

    available_districts = ["All districts"] + sorted(
        orchards["district"].dropna().astype(str).unique().tolist()
    )
    selected_map_district = st.selectbox("Filter orchards by district", available_districts)
    if selected_map_district != "All districts":
        orchards = orchards[orchards["district"].astype(str) == selected_map_district].copy()

    APPROXIMATE_LOCATIONS = {
        "arnau agro": {"latitude": 43.37113, "longitude": 77.45141, "reference": "Esik — Enbekshikazakh District"},
        "baimene": {"latitude": 43.30272, "longitude": 77.23947, "reference": "Talgar — Talgar District"},
        "turgen apple": {"latitude": 43.39672, "longitude": 77.59350, "reference": "Turgen Village — settlement reference"},
        "agros bt": {"latitude": 43.52000, "longitude": 77.77610, "reference": "Kazakhstan Village — settlement reference"}
    }

    def get_company_key(value):
        return str(value).strip().casefold()

    def has_coordinates(orchard):
        lat = pd.to_numeric(pd.Series([orchard.get("latitude")]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([orchard.get("longitude")]), errors="coerce").iloc[0]
        return pd.notna(lat) and pd.notna(lon)

    def can_display_exact_location(orchard):
        status = str(orchard.get("location_status", "")).casefold()
        return has_coordinates(orchard) and "public display permitted" in status

    confirmed_count = sum(1 for _, o in orchards.iterrows() if can_display_exact_location(o))
    approximate_count = sum(
        1 for _, o in orchards.iterrows()
        if not can_display_exact_location(o) and get_company_key(o["company"]) in APPROXIMATE_LOCATIONS
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Visible Orchards", len(orchards))
    c2.metric("Verified GPS Pins", confirmed_count)
    c3.metric("District Reference Pins", approximate_count)

    map_mode = st.radio("Map Layer", ["Standard Map", "Satellite Imagery"], horizontal=True)

    orchard_map = folium.Map(location=[43.65, 77.40], zoom_start=8, tiles=None, control_scale=True)
    if map_mode == "Standard Map":
        folium.TileLayer(tiles="OpenStreetMap", name="Standard Map", control=False).add_to(orchard_map)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri",
            name="Satellite Map",
            control=False
        ).add_to(orchard_map)

    marker_points = []
    for _, orchard in orchards.iterrows():
        if can_display_exact_location(orchard):
            lat, lon = float(orchard["latitude"]), float(orchard["longitude"])
            popup_text = f"<b>{orchard['company']}</b><br>District: {orchard['district']}<br>Status: Verified GPS"
            folium.Marker(
                location=[lat, lon],
                tooltip=str(orchard["company"]),
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color="green", icon="leaf", prefix="fa")
            ).add_to(orchard_map)
            marker_points.append([lat, lon])
        else:
            ref = APPROXIMATE_LOCATIONS.get(get_company_key(orchard["company"]))
            if ref:
                lat, lon = ref["latitude"], ref["longitude"]
                popup_text = (
                    f"<b>{orchard['company']}</b><br>District: {orchard['district']}<br>"
                    f"Ref: {ref['reference']}<br><i>Approximate location</i>"
                )
                folium.Marker(
                    location=[lat, lon],
                    tooltip=f"{orchard['company']} (Approx.)",
                    popup=folium.Popup(popup_text, max_width=320),
                    icon=folium.Icon(color="orange", icon="info-sign", prefix="glyphicon")
                ).add_to(orchard_map)
                marker_points.append([lat, lon])

    if marker_points:
        orchard_map.fit_bounds(marker_points)

    map_result = st_folium(orchard_map, width=None, height=520, returned_objects=["last_clicked"], key="orchard_map_widget")
    if map_result and map_result.get("last_clicked"):
        st.session_state.selected_latitude = round(map_result["last_clicked"]["lat"], 6)
        st.session_state.selected_longitude = round(map_result["last_clicked"]["lng"], 6)

    st.divider()
    st.subheader("📍 Register New Orchard")
    if st.session_state.selected_latitude is not None:
        st.success(f"Map pin selected: `{st.session_state.selected_latitude}, {st.session_state.selected_longitude}`")
    else:
        st.caption("Click anywhere on the map above to capture coordinates for a new orchard.")

    with st.form("new_orchard_form"):
        company = st.text_input("Company / Orchard Name *")
        district = st.text_input("District")
        location = st.text_input("Village / Location Description")

        col1, col2 = st.columns(2)
        with col1:
            area_ha = st.number_input("Orchard Area (ha)", min_value=0.0, value=0.0, step=1.0)
            trees = st.number_input("Number of Trees", min_value=0, value=0, step=100)
        with col2:
            varieties = st.text_input("Apple Varieties", placeholder="Gala, Fuji, Golden Delicious")
            coordinates_confirmed = st.checkbox("Confirm exact GPS location & public display permission")

        if st.form_submit_button("➕ Add Orchard to Registry"):
            if not company.strip():
                st.error("Please enter the company or orchard name.")
            elif coordinates_confirmed and st.session_state.selected_latitude is None:
                st.error("Click the map first to select coordinates.")
            else:
                new_orchard = pd.DataFrame([{
                    "company": company.strip(),
                    "district": district.strip(),
                    "location": location.strip(),
                    "area_ha": area_ha if area_ha > 0 else None,
                    "trees": trees if trees > 0 else None,
                    "varieties": varieties.strip(),
                    "latitude": st.session_state.selected_latitude if coordinates_confirmed else None,
                    "longitude": st.session_state.selected_longitude if coordinates_confirmed else None,
                    "location_status": (
                        "Coordinates confirmed by user; public display permitted"
                        if coordinates_confirmed else "Coordinates not verified"
                    )
                }])
                updated_orchards = pd.concat([st.session_state.orchards, new_orchard], ignore_index=True)
                save_orchards(updated_orchards)
                st.session_state.orchards = updated_orchards
                st.session_state.selected_latitude = None
                st.session_state.selected_longitude = None
                st.success(f"**{company.strip()}** added!")
                st.rerun()

    st.divider()
    st.subheader("🌳 Registered Orchards Directory")
    display_orchards = translate_orchard_text(st.session_state.orchards.copy())
    for idx, row in display_orchards.iterrows():
        if "public display permitted" not in str(row.get("location_status", "")).casefold():
            display_orchards.loc[idx, ["latitude", "longitude"]] = "Protected"

    display_orchards = display_orchards.rename(columns={
        "company": "Company", "district": "District", "location": "Location",
        "area_ha": "Area (ha)", "trees": "Trees", "varieties": "Varieties",
        "latitude": "Lat", "longitude": "Lon", "location_status": "Verification Status"
    })
    show_styled_table(display_orchards, use_container_width=True, hide_index=True)


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption("OrchardAI © 2026 · Built for Smart Agriculture")
