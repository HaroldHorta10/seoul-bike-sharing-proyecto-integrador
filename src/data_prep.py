"""
Pipeline de carga, limpieza, ingeniería de variables y partición de datos
para el dataset Seoul Bike Sharing Demand.

Replica exactamente las decisiones justificadas en notebooks/01 y notebooks/02:
- Excluir horas con Functioning_Day == "No" (demanda forzada a 0, no es señal real).
- Transformación log1p sobre Rented_Bike_Count, Rainfall y Snowfall (asimetría alta).
- Variables derivadas: Mes, Es_Fin_de_Semana, Franja_Horaria.
- Partición 70/15/15 estratificada por Seasons.
- Dos conjuntos de features INDEPENDIENTES (regresión vs. clasificación) para
  evitar que el target de una tarea se filtre como feature de la otra.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Relativo a este archivo, no al cwd del proceso: streamlit run (incluido
# Streamlit Community Cloud) y los notebooks no siempre arrancan con el mismo
# directorio de trabajo.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(_SRC_DIR, "..", "data", "raw", "SeoulBikeData.csv")

CAT_COLS_REG = ["Seasons", "Holiday", "Franja_Horaria"]
NUM_COLS_REG = ["Hour", "Temperature", "Humidity", "Wind_speed", "Visibility",
                "Dew_point_temp", "Solar_Radiation", "log_Rainfall", "log_Snowfall",
                "Mes", "Es_Fin_de_Semana"]

CAT_COLS_CLF = ["Holiday", "Franja_Horaria"]
# Nota: "Mes" se excluye deliberadamente. A diferencia de Seasons vs. demanda
# (correlación), Mes determina a Seasons de forma 1 a 1 en este dataset (es la
# regla con la que se construyó la columna) -> incluirlo sería la misma fuga
# de datos que ya corregimos, solo que por una puerta trasera.
NUM_COLS_CLF = ["Hour", "Temperature", "Humidity", "Wind_speed", "Visibility",
                "Dew_point_temp", "Solar_Radiation", "log_Rainfall", "log_Snowfall",
                "Es_Fin_de_Semana", "log_Rented_Bike_Count"]

CLIMA_COLS_CLUSTER = ["Temperature", "Humidity", "Wind_speed", "Visibility",
                      "Dew_point_temp", "Solar_Radiation", "log_Rainfall", "log_Snowfall"]


def _cargar_y_limpiar(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path, encoding="cp1252")
    df = df.rename(columns={
        "Rented Bike Count": "Rented_Bike_Count",
        "Temperature(°C)": "Temperature",
        "Humidity(%)": "Humidity",
        "Wind speed (m/s)": "Wind_speed",
        "Visibility (10m)": "Visibility",
        "Dew point temperature(°C)": "Dew_point_temp",
        "Solar Radiation (MJ/m2)": "Solar_Radiation",
        "Rainfall(mm)": "Rainfall",
        "Snowfall (cm)": "Snowfall",
        "Functioning Day": "Functioning_Day",
    })

    df_limpio = df[df["Functioning_Day"] == "Yes"].copy()
    df_limpio = df_limpio.drop(columns=["Functioning_Day"])

    for col in ["Rented_Bike_Count", "Rainfall", "Snowfall"]:
        df_limpio[f"log_{col}"] = np.log1p(df_limpio[col])

    df_limpio["Date"] = pd.to_datetime(df_limpio["Date"], format="%d/%m/%Y")
    df_limpio["Mes"] = df_limpio["Date"].dt.month
    df_limpio["Es_Fin_de_Semana"] = df_limpio["Date"].dt.day_of_week.isin([5, 6]).astype(int)

    def franja(h):
        if 5 <= h < 12:
            return "Mañana"
        if 12 <= h < 18:
            return "Tarde"
        if 18 <= h < 23:
            return "Noche"
        return "Madrugada"

    df_limpio["Franja_Horaria"] = df_limpio["Hour"].apply(franja)
    return df_limpio


@dataclass
class DatasetPreparado:
    df_limpio: pd.DataFrame
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame

    X_train_reg: pd.DataFrame
    X_val_reg: pd.DataFrame
    X_test_reg: pd.DataFrame
    y_reg_train: pd.Series
    y_reg_val: pd.Series
    y_reg_test: pd.Series

    X_train_clf: pd.DataFrame
    X_val_clf: pd.DataFrame
    X_test_clf: pd.DataFrame
    y_clf_train: pd.Series
    y_clf_val: pd.Series
    y_clf_test: pd.Series

    encoder_reg: OneHotEncoder
    scaler_reg: StandardScaler
    encoder_clf: OneHotEncoder
    scaler_clf: StandardScaler


def preparar_dataset(raw_path: str = RAW_PATH, random_state: int = 42) -> DatasetPreparado:
    df_limpio = _cargar_y_limpiar(raw_path)

    train_idx, temp_idx = train_test_split(
        df_limpio.index, test_size=0.30, random_state=random_state, stratify=df_limpio["Seasons"]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=random_state,
        stratify=df_limpio.loc[temp_idx, "Seasons"]
    )
    df_train = df_limpio.loc[train_idx]
    df_val = df_limpio.loc[val_idx]
    df_test = df_limpio.loc[test_idx]

    encoder_reg = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder_reg.fit(df_train[CAT_COLS_REG])
    scaler_reg = StandardScaler()
    scaler_reg.fit(df_train[NUM_COLS_REG])

    def transformar_reg(df_split):
        cat_arr = encoder_reg.transform(df_split[CAT_COLS_REG])
        cat_df = pd.DataFrame(cat_arr, columns=encoder_reg.get_feature_names_out(CAT_COLS_REG), index=df_split.index)
        num_arr = scaler_reg.transform(df_split[NUM_COLS_REG])
        num_df = pd.DataFrame(num_arr, columns=NUM_COLS_REG, index=df_split.index)
        return pd.concat([num_df, cat_df], axis=1)

    encoder_clf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder_clf.fit(df_train[CAT_COLS_CLF])
    scaler_clf = StandardScaler()
    scaler_clf.fit(df_train[NUM_COLS_CLF])

    def transformar_clf(df_split):
        cat_arr = encoder_clf.transform(df_split[CAT_COLS_CLF])
        cat_df = pd.DataFrame(cat_arr, columns=encoder_clf.get_feature_names_out(CAT_COLS_CLF), index=df_split.index)
        num_arr = scaler_clf.transform(df_split[NUM_COLS_CLF])
        num_df = pd.DataFrame(num_arr, columns=NUM_COLS_CLF, index=df_split.index)
        return pd.concat([num_df, cat_df], axis=1)

    return DatasetPreparado(
        df_limpio=df_limpio, df_train=df_train, df_val=df_val, df_test=df_test,
        X_train_reg=transformar_reg(df_train), X_val_reg=transformar_reg(df_val), X_test_reg=transformar_reg(df_test),
        y_reg_train=df_train["log_Rented_Bike_Count"], y_reg_val=df_val["log_Rented_Bike_Count"], y_reg_test=df_test["log_Rented_Bike_Count"],
        X_train_clf=transformar_clf(df_train), X_val_clf=transformar_clf(df_val), X_test_clf=transformar_clf(df_test),
        y_clf_train=df_train["Seasons"], y_clf_val=df_val["Seasons"], y_clf_test=df_test["Seasons"],
        encoder_reg=encoder_reg, scaler_reg=scaler_reg, encoder_clf=encoder_clf, scaler_clf=scaler_clf,
    )
