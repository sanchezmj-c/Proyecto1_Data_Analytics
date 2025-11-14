import os
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Streaming Anomalies – EIA",
    layout="wide",
)

st.title("⚡ Streaming de Anomalías en Intercambio de Energía (Demo)")

# ----------------------------------------------------
# CONFIG PARA FUTURO AZURE POSTGRES (AÚN SIN USAR)
# ----------------------------------------------------
PG_HOST = os.getenv("PG_HOST", "TU_SERVIDOR.postgres.database.azure.com")
PG_DB   = os.getenv("PG_DB", "tu_base")
PG_USER = os.getenv("PG_USER", "tu_usuario")
PG_PWD  = os.getenv("PG_PWD", "tu_password")
PG_PORT = os.getenv("PG_PORT", "5432")

# ----------------------------------------------------
# 1. GENERACIÓN DE DATOS SINTÉTICOS (INSPIRADO EN TU NOTEBOOK)
# ----------------------------------------------------
def generate_synthetic_data(
    n_records: int = 50_000,
    start_date: str = "2025-10-01",
    end_date: str = "2025-11-10",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera datos sintéticos realistas de intercambio de energía entre BAs.

    Esto está inspirado en lo que haces en el notebook:
    - BAs reales de EIA
    - Rutas fromba → toba
    - Patrón horario (día/noche) + ruido
    - Algunos duplicados y nulos en value-units
    """
    np.random.seed(seed)

    # Balancing Authorities (como en tu notebook)
    bas = [
        "BPAT", "IPCO", "PACW", "PACE", "NEVP",
        "NWMT", "PGE", "PSCO", "AZPS", "SRP",
        "WALC",
    ]

    # Todas las rutas posibles from → to (sin self-loop)
    routes = [(f, t) for f in bas for t in bas if f != t]

    # Rango temporal horario
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    timestamps = pd.date_range(start=start, end=end, freq="h")

    # Elegimos timestamps y rutas al azar como si fueran "eventos" de streaming
    ts_idx = np.random.choice(len(timestamps), size=n_records, replace=True)
    route_idx = np.random.choice(len(routes), size=n_records, replace=True)

    periods = timestamps[ts_idx]
    fromba = [routes[i][0] for i in route_idx]
    toba   = [routes[i][1] for i in route_idx]

    # Patrón base diario (pico en ciertas horas)
    hours = np.array([p.hour for p in periods])
    # Componente sinusoidal diaria
    daily_pattern = 1000 + 400 * np.sin(2 * np.pi * hours / 24)

    # Efecto día de la semana (ligero)
    dow = np.array([p.weekday() for p in periods])
    weekday_factor = np.where(dow < 5, 1.0, 0.8)  # menos flujo en fines de semana

    # Ruido aleatorio
    noise = np.random.normal(loc=0, scale=120, size=n_records)

    values = daily_pattern * weekday_factor + noise

    # Asegurar valores positivos en general (pero permitimos algo de variabilidad)
    values = np.clip(values, a_min=50, a_max=None)

    # Construir DataFrame
    df = pd.DataFrame(
        {
            "period": periods,
            "fromba": fromba,
            "toba": toba,
            "value": values,
            "value-units": "MW",
        }
    )

    # Añadir algunos duplicados intencionales (~0.5%) como en tu notebook
    n_duplicates = int(len(df) * 0.005)
    if n_duplicates > 0:
        dup_idx = np.random.choice(df.index, n_duplicates, replace=False)
        df_dups = df.loc[dup_idx]
        df = pd.concat([df, df_dups], ignore_index=True)

    # Añadir nulos en 'value-units' (~10%, columna no crítica)
    n_nulls = int(len(df) * 0.10)
    null_idx = np.random.choice(df.index, n_nulls, replace=False)
    df.loc[null_idx, "value-units"] = None

    return df


# ----------------------------------------------------
# 2. INGESTA DE DATOS
#    (HOY: SINTÉTICO; FUTURO: POSTGRES)
# ----------------------------------------------------
@st.cache_data(show_spinner=True, ttl=5)
def load_from_synthetic(
    n_records: int = 50_000,
    start_date: str = "2025-10-01",
    end_date: str = "2025-11-10",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera datos sintéticos "como el notebook".
    Se vuelve a generar (simular streaming) cada vez que cambias algo o pulsas actualizar.
    """
    df = generate_synthetic_data(
        n_records=n_records,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )
    return df


@st.cache_data(show_spinner=True, ttl=5)
def load_from_postgres() -> pd.DataFrame:
    """
    FUTURO: leer desde Azure Postgres.
    Ahora mismo devuelve df vacío + mensaje.
    Cuando tengas la tabla, SOLO toca esta función.
    """
    try:
        url = (
            f"postgresql+psycopg2://{PG_USER}:{PG_PWD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
            "?sslmode=require"
        )
        engine = create_engine(url)

        # TODO: cuando tengas la tabla de streaming, pon aquí tu query:
        # query = '''
        # SELECT period, fromba, toba, value, "value-units"
        # FROM eia_interchange_stream
        # WHERE period >= NOW() - interval '7 days';
        # '''
        # df = pd.read_sql(query, engine)
        # return df

        st.info("🔧 Conexión a Postgres aún no configurada. Edita load_from_postgres() cuando tengas la tabla.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error conectando a Postgres: {e}")
        return pd.DataFrame()


# ----------------------------------------------------
# 3. LIMPIEZA + FEATURES (RESUMEN DE LO DEL NOTEBOOK)
# ----------------------------------------------------
def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    # period
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df = df.dropna(subset=["period"])

    # value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # normalizar fromba/toba
    for col in ["fromba", "toba"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
        else:
            df[col] = None

    # route
    df["route"] = df["fromba"].fillna("?") + " → " + df["toba"].fillna("?")

    # features temporales
    df["hour"] = df["period"].dt.hour
    df["day_of_week"] = df["period"].dt.dayofweek
    df["month"] = df["period"].dt.month
    df["year"] = df["period"].dt.year

    df = df.sort_values("period").reset_index(drop=True)

    # diferencias por ruta (para anomalía temporal)
    df["value_diff"] = df.groupby("route")["value"].diff()
    df["value_pct_change"] = df.groupby("route")["value"].pct_change()

    return df


# ----------------------------------------------------
# 4. DETECCIÓN DE ANOMALÍAS (INSPIRADO EN TU NOTEBOOK)
#    AQUÍ LUEGO PUEDES PEGAR TU LÓGICA EXACTA
# ----------------------------------------------------
def detect_anomalies(df_clean: pd.DataFrame) -> pd.DataFrame:
    if df_clean.empty:
        return df_clean

    df = df_clean.copy()

    # --- Z-SCORE ---
    mu = df["value"].mean()
    sigma = df["value"].std(ddof=0) or 1.0
    df["z_score"] = (df["value"] - mu) / sigma
    df["anomaly_zscore"] = df["z_score"].abs() > 3

    # --- IQR ---
    q1 = df["value"].quantile(0.25)
    q3 = df["value"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df["anomaly_iqr"] = (df["value"] < lower) | (df["value"] > upper)

    # --- Isolation Forest (multivariado ligero) ---
    features = ["value", "hour", "day_of_week"]
    for f in features:
        if f not in df.columns:
            df[f] = 0
    X = df[features].fillna(0)

    iso = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=100,
    )
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)

    df["anomaly_score_if"] = scores
    df["anomaly_if"] = preds == -1

    # --- Anomalía temporal simple ---
    df["anomaly_temporal"] = df["value_diff"].abs() > (df["value"].std() * 2)

    # --- Combinado ---
    df["anomaly_count"] = (
        df["anomaly_zscore"].astype(int)
        + df["anomaly_iqr"].astype(int)
        + df["anomaly_if"].astype(int)
        + df["anomaly_temporal"].astype(int)
    )
    df["anomaly_combined"] = df["anomaly_count"] >= 2

    return df


# ----------------------------------------------------
# 5. SIDEBAR – CONTROLES DE STREAMING
# ----------------------------------------------------
st.sidebar.header("🎛 Configuración de streaming")

source = st.sidebar.radio(
    "Fuente de datos",
    options=["Sintético (como notebook)", "Azure Postgres (futuro)"],
    index=0,
)

# Parámetros del generador (solo sintético)
if source == "Sintético (como notebook)":
    n_records = st.sidebar.slider("Nº de registros sintéticos", 5_000, 100_000, 50_000, step=5_000)
    start_date = st.sidebar.date_input("Fecha inicial", date(2025, 10, 1))
    end_date = st.sidebar.date_input("Fecha final", date(2025, 11, 10))
    seed = st.sidebar.number_input("Seed aleatoria", min_value=0, max_value=1_000_000, value=42, step=1)
else:
    # Para Postgres no pedimos parámetros por ahora
    n_records = None
    start_date = None
    end_date = None
    seed = None

if st.sidebar.button("🔄 Actualizar ahora (simular streaming)"):
    st.experimental_rerun()

st.sidebar.caption("Cada actualización vuelve a leer/generar los datos y recalcula anomalías.")


# ----------------------------------------------------
# 6. INGESTA + PIPELINE (FUENTE → CLEAN → ANOMALÍAS)
# ----------------------------------------------------
if source == "Sintético (como notebook)":
    df_raw = load_from_synthetic(
        n_records=n_records,
        start_date=str(start_date),
        end_date=str(end_date),
        seed=seed,
    )
else:
    df_raw = load_from_postgres()

if df_raw.empty:
    st.warning("No hay datos disponibles todavía.")
    st.stop()

df_clean = clean_data(df_raw)
df_proc = detect_anomalies(df_clean)

# ----------------------------------------------------
# 7. FILTROS (TIEMPO, RUTA, MÉTODOS)
# ----------------------------------------------------
st.subheader("📦 Datos procesados después de limpieza + anomalías")

min_date = df_proc["period"].min().date()
max_date = df_proc["period"].max().date()

col_f1, col_f2 = st.columns(2)
with col_f1:
    date_range = st.date_input(
        "Rango de fechas (visualización)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(date_range, tuple):
    start_viz, end_viz = date_range
else:
    start_viz, end_viz = min_date, max_date

mask_date = (df_proc["period"].dt.date >= start_viz) & (df_proc["period"].dt.date <= end_viz)
df_filtered = df_proc.loc[mask_date].copy()

routes = sorted(df_filtered["route"].dropna().unique())
with col_f2:
    route_sel = st.selectbox("Ruta (from → to)", options=["(Todas)"] + routes, index=0)

if route_sel != "(Todas)":
    df_filtered = df_filtered[df_filtered["route"] == route_sel]

# Métodos de anomalía
st.sidebar.markdown("### Métodos de anomalía activos")
anomaly_methods = [
    ("anomaly_zscore", "Z-Score"),
    ("anomaly_iqr", "IQR"),
    ("anomaly_if", "Isolation Forest"),
    ("anomaly_temporal", "Temporal"),
]

selected_cols = []
for col, label in anomaly_methods:
    if col in df_filtered.columns:
        if st.sidebar.checkbox(label, value=True):
            selected_cols.append(col)

if selected_cols:
    df_filtered["anomaly_custom"] = df_filtered[selected_cols].sum(axis=1) >= 1
else:
    df_filtered["anomaly_custom"] = df_filtered["anomaly_combined"]

# ----------------------------------------------------
# 8. KPIs
# ----------------------------------------------------
total_reg = len(df_filtered)
total_anom = df_filtered["anomaly_custom"].sum()
pct_anom = (total_anom / total_reg * 100) if total_reg > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros (ventana)", f"{total_reg:,}")
col2.metric("Anomalías (ventana)", f"{total_anom:,}", f"{pct_anom:.2f}%")
col3.metric("Rutas únicas", f"{df_filtered['route'].nunique():,}")
col4.metric(
    "Regiones from/to",
    f"{df_filtered['fromba'].nunique()} / {df_filtered['toba'].nunique()}",
)

st.markdown("---")

# ----------------------------------------------------
# 9. TABS DEL DASHBOARD
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔭 Visión general",
    "⏱ Patrones temporales",
    "🧭 Rutas",
    "🧪 Tabla de anomalías",
])

# ---- TAB 1: Serie temporal
with tab1:
    st.subheader("Serie temporal con anomalías resaltadas")

    df_plot = df_filtered.sort_values("period")
    df_plot["is_anomaly"] = df_plot["anomaly_custom"].astype(bool)

    base = alt.Chart(df_plot).encode(
        x=alt.X("period:T", title="Fecha/Hora"),
        y=alt.Y("value:Q", title="Valor (MW)"),
    )

    line = base.mark_line().encode(
        tooltip=["period:T", "value:Q", "fromba", "toba", "route"]
    )

    points = base.transform_filter(
        alt.datum.is_anomaly == True
    ).mark_circle(size=60).encode(
        color=alt.value("red"),
        tooltip=[
            "period:T",
            "value:Q",
            "fromba",
            "toba",
            "route",
            "z_score",
            "anomaly_zscore",
            "anomaly_iqr",
            "anomaly_if",
            "anomaly_temporal",
            "anomaly_combined",
        ],
    )

    chart = (line + points).interactive().properties(height=420)
    st.altair_chart(chart, use_container_width=True)

# ---- TAB 2: Patrones temporales
with tab2:
    st.subheader("Patrones por hora y día de la semana")

    if "hour" not in df_filtered.columns:
        df_filtered["hour"] = df_filtered["period"].dt.hour
    if "day_of_week" not in df_filtered.columns:
        df_filtered["day_of_week"] = df_filtered["period"].dt.dayofweek

    hourly = (
        df_filtered.groupby("hour")["value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    st.markdown("#### Valor medio por hora del día")
    chart_hour = (
        alt.Chart(hourly)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:O", title="Hora del día"),
            y=alt.Y("mean:Q", title="Valor medio (MW)"),
            size=alt.Size("count:Q", title="Nº registros"),
            tooltip=["hour", "mean", "count"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart_hour, use_container_width=True)

    dow = (
        df_filtered.groupby("day_of_week")["value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    dow_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dow["day_name"] = dow["day_of_week"].map({i: n for i, n in enumerate(dow_names)})

    st.markdown("#### Valor medio por día de la semana")
    chart_dow = (
        alt.Chart(dow)
        .mark_bar()
        .encode(
            x=alt.X("day_name:N", title="Día", sort=dow_names),
            y=alt.Y("mean:Q", title="Valor medio (MW)"),
            tooltip=["day_name", "mean", "count"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart_dow, use_container_width=True)

# ---- TAB 3: Rutas
with tab3:
    st.subheader("Rutas más activas y con más anomalías")

    route_stats = (
        df_filtered
        .groupby("route")
        .agg(
            total_value=("value", "sum"),
            mean_value=("value", "mean"),
            count=("value", "count"),
            anomalies=("anomaly_custom", "sum"),
        )
        .reset_index()
    )

    top_n = st.slider("Top N rutas", 5, 30, 10)

    st.markdown("#### Top rutas por valor total")
    top_routes = route_stats.sort_values("total_value", ascending=False).head(top_n)

    chart_routes = (
        alt.Chart(top_routes)
        .mark_bar()
        .encode(
            x=alt.X("total_value:Q", title="Total (MW)"),
            y=alt.Y("route:N", sort="-x", title="Ruta"),
            color=alt.Color("anomalies:Q", title="Nº anomalías"),
            tooltip=["route", "total_value", "mean_value", "count", "anomalies"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart_routes, use_container_width=True)

    st.markdown("#### Rutas con más anomalías")
    top_anom = route_stats.sort_values("anomalies", ascending=False).head(top_n)
    st.dataframe(top_anom)

# ---- TAB 4: Tabla de anomalías
with tab4:
    st.subheader("Detalle de anomalías (ventana filtrada)")

    df_anom = df_filtered[df_filtered["anomaly_custom"]].copy()

    if df_anom.empty:
        st.info("No hay anomalías con los filtros actuales.")
    else:
        sort_cols = []
        if "anomaly_score_if" in df_anom.columns:
            sort_cols = ["anomaly_score_if"]
        elif "value_diff" in df_anom.columns:
            sort_cols = ["value_diff"]
        else:
            sort_cols = ["period"]

        df_anom = df_anom.sort_values(sort_cols, ascending=False)

        cols_show = [
            c
            for c in [
                "period", "fromba", "toba", "route",
                "value", "value_diff", "value_pct_change",
                "z_score", "anomaly_zscore", "anomaly_iqr",
                "anomaly_if", "anomaly_temporal", "anomaly_combined",
                "anomaly_score_if",
            ]
            if c in df_anom.columns
        ]

        st.dataframe(df_anom[cols_show].head(500))
        st.caption("Se muestran hasta 500 filas.")
