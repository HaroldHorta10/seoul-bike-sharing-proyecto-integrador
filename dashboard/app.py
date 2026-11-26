"""
Dashboard interactivo — Proyecto Integrador de Ciencia de Datos
Seoul Bike Sharing Demand

Ejecutar con:  streamlit run app.py   (desde la carpeta /dashboard, con el venv activo)
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from data_prep import CLIMA_COLS_CLUSTER, preparar_dataset  # noqa: E402

st.set_page_config(page_title="Seoul Bike Sharing — Dashboard", layout="wide")


@st.cache_data
def cargar_datos():
    d = preparar_dataset()
    return d.df_limpio


@st.cache_resource
def cargar_modelos():
    return {
        "regresion": joblib.load(os.path.join(MODELS_DIR, "modelo_regresion.joblib")),
        "clasificacion": joblib.load(os.path.join(MODELS_DIR, "modelo_clasificacion.joblib")),
        "kmeans": joblib.load(os.path.join(MODELS_DIR, "modelo_kmeans.joblib")),
        "scaler_cluster": joblib.load(os.path.join(MODELS_DIR, "scaler_cluster.joblib")),
        "pca": joblib.load(os.path.join(MODELS_DIR, "pca_cluster.joblib")),
        "encoder_reg": joblib.load(os.path.join(MODELS_DIR, "encoder_reg.joblib")),
        "scaler_reg": joblib.load(os.path.join(MODELS_DIR, "scaler_reg.joblib")),
    }


df = cargar_datos()
modelos = cargar_modelos()

st.title("🚲 Seoul Bike Sharing — Dashboard Interactivo")
st.caption("Proyecto Integrador de Ciencia de Datos — Universidad de La Salle, Maestría en IA")

# ---------------------------------------------------------------------------
# Filtros dinámicos (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")

fecha_min, fecha_max = df["Date"].min(), df["Date"].max()
rango_fechas = st.sidebar.date_input(
    "Rango de fechas", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max
)

estaciones_sel = st.sidebar.multiselect(
    "Estación del año", options=sorted(df["Seasons"].unique()), default=sorted(df["Seasons"].unique())
)

temp_min, temp_max = float(df["Temperature"].min()), float(df["Temperature"].max())
rango_temp = st.sidebar.slider(
    "Temperatura (°C)", min_value=temp_min, max_value=temp_max, value=(temp_min, temp_max)
)

holiday_sel = st.sidebar.multiselect(
    "Tipo de día", options=sorted(df["Holiday"].unique()), default=sorted(df["Holiday"].unique())
)

if len(rango_fechas) == 2:
    f_ini, f_fin = pd.Timestamp(rango_fechas[0]), pd.Timestamp(rango_fechas[1])
else:
    f_ini, f_fin = fecha_min, fecha_max

df_filtrado = df[
    (df["Date"] >= f_ini) & (df["Date"] <= f_fin)
    & (df["Seasons"].isin(estaciones_sel))
    & (df["Temperature"].between(*rango_temp))
    & (df["Holiday"].isin(holiday_sel))
]

if df_filtrado.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada. Ajustá los filtros en la barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# Tarjetas KPI
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Horas en el filtro", f"{len(df_filtrado):,}")
col2.metric("Demanda promedio/hora", f"{df_filtrado['Rented_Bike_Count'].mean():.0f}")
col3.metric("Demanda pico (1 hora)", f"{df_filtrado['Rented_Bike_Count'].max():,}")
hora_pico = df_filtrado.groupby("Hour")["Rented_Bike_Count"].mean().idxmax()
col4.metric("Hora de mayor demanda promedio", f"{hora_pico}:00")

st.divider()

tab_resumen, tab_prediccion, tab_clustering = st.tabs(
    ["📊 Resumen", "🔮 Predicción en vivo", "🧩 Segmentación (Clustering)"]
)

# ---------------------------------------------------------------------------
# Tab 1: Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    c1, c2 = st.columns(2)
    with c1:
        demanda_por_hora = df_filtrado.groupby("Hour")["Rented_Bike_Count"].mean().reset_index()
        fig1 = px.line(demanda_por_hora, x="Hour", y="Rented_Bike_Count",
                        title="Demanda promedio por hora del día", markers=True)
        st.plotly_chart(fig1, width="stretch")
    with c2:
        demanda_por_estacion = df_filtrado.groupby("Seasons")["Rented_Bike_Count"].mean().reset_index()
        fig2 = px.bar(demanda_por_estacion, x="Seasons", y="Rented_Bike_Count",
                       title="Demanda promedio por estación")
        st.plotly_chart(fig2, width="stretch")

    fig3 = px.scatter(df_filtrado, x="Temperature", y="Rented_Bike_Count", color="Seasons",
                       opacity=0.4, title="Temperatura vs. demanda horaria (datos filtrados)")
    st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: Predicción en vivo
# ---------------------------------------------------------------------------
with tab_prediccion:
    st.subheader("Simulador de demanda horaria")
    st.caption("Ingresá condiciones climáticas y temporales para obtener una predicción instantánea del modelo de regresión (Random Forest, R² = 0.915 en test).")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        hour_in = st.slider("Hora del día", 0, 23, 8)
        temp_in = st.slider("Temperatura (°C)", -20.0, 40.0, 15.0)
        humidity_in = st.slider("Humedad (%)", 0, 100, 50)
    with pc2:
        wind_in = st.slider("Velocidad del viento (m/s)", 0.0, 8.0, 1.5)
        visibility_in = st.slider("Visibilidad (10m)", 0, 2000, 1500)
        dew_in = st.slider("Punto de rocío (°C)", -30.0, 30.0, 5.0)
    with pc3:
        solar_in = st.slider("Radiación solar (MJ/m²)", 0.0, 4.0, 0.5)
        rainfall_in = st.slider("Precipitación (mm)", 0.0, 35.0, 0.0)
        snowfall_in = st.slider("Nevada (cm)", 0.0, 9.0, 0.0)

    season_in = st.selectbox("Estación del año", sorted(df["Seasons"].unique()))
    holiday_in = st.selectbox("¿Es festivo?", sorted(df["Holiday"].unique()))

    def franja(h):
        if 5 <= h < 12:
            return "Mañana"
        if 12 <= h < 18:
            return "Tarde"
        if 18 <= h < 23:
            return "Noche"
        return "Madrugada"

    mes_in = st.selectbox("Mes", list(range(1, 13)), index=5)
    es_finde_in = st.checkbox("¿Es fin de semana?")

    if st.button("Predecir demanda", type="primary"):
        fila = pd.DataFrame([{
            "Hour": hour_in, "Temperature": temp_in, "Humidity": humidity_in,
            "Wind_speed": wind_in, "Visibility": visibility_in, "Dew_point_temp": dew_in,
            "Solar_Radiation": solar_in,
            "log_Rainfall": np.log1p(rainfall_in), "log_Snowfall": np.log1p(snowfall_in),
            "Mes": mes_in, "Es_Fin_de_Semana": int(es_finde_in),
            "Seasons": season_in, "Holiday": holiday_in, "Franja_Horaria": franja(hour_in),
        }])

        cat_cols_reg = ["Seasons", "Holiday", "Franja_Horaria"]
        num_cols_reg = ["Hour", "Temperature", "Humidity", "Wind_speed", "Visibility",
                         "Dew_point_temp", "Solar_Radiation", "log_Rainfall", "log_Snowfall",
                         "Mes", "Es_Fin_de_Semana"]
        cat_arr = modelos["encoder_reg"].transform(fila[cat_cols_reg])
        cat_df = pd.DataFrame(cat_arr, columns=modelos["encoder_reg"].get_feature_names_out(cat_cols_reg))
        num_arr = modelos["scaler_reg"].transform(fila[num_cols_reg])
        num_df = pd.DataFrame(num_arr, columns=num_cols_reg)
        X_pred = pd.concat([num_df, cat_df], axis=1)

        log_pred = modelos["regresion"].predict(X_pred)[0]
        pred_bicis = np.expm1(log_pred)
        st.success(f"**Demanda estimada: {pred_bicis:,.0f} bicicletas en esa hora** (predicción revertida de escala logarítmica)")

# ---------------------------------------------------------------------------
# Tab 3: Clustering
# ---------------------------------------------------------------------------
with tab_clustering:
    st.subheader("Perfiles climáticos (K-Means, K=5)")
    st.caption("Segmentación no supervisada sobre variables climáticas estandarizadas, proyectada en espacio PCA 2D.")

    X_clima_filtrado = modelos["scaler_cluster"].transform(df_filtrado[CLIMA_COLS_CLUSTER])
    clusters_filtrado = modelos["kmeans"].predict(X_clima_filtrado)
    X_pca_filtrado = modelos["pca"].transform(X_clima_filtrado)

    df_plot = pd.DataFrame({
        "PC1": X_pca_filtrado[:, 0], "PC2": X_pca_filtrado[:, 1],
        "Cluster": clusters_filtrado.astype(str),
        "Demanda": df_filtrado["Rented_Bike_Count"].values,
    })
    fig_cluster = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
                              hover_data=["Demanda"], opacity=0.5,
                              title="Clusters climáticos en espacio PCA (datos filtrados)")
    st.plotly_chart(fig_cluster, width="stretch")

    df_filtrado_cl = df_filtrado.copy()
    df_filtrado_cl["cluster"] = clusters_filtrado
    perfiles = df_filtrado_cl.groupby("cluster")[CLIMA_COLS_CLUSTER + ["Rented_Bike_Count"]].mean().round(2)
    perfiles["n_obs"] = df_filtrado_cl.groupby("cluster").size()
    st.dataframe(perfiles, width="stretch")
