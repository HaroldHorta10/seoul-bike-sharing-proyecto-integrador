"""
Entrena los modelos ganadores de la Fase 3 y los persiste con joblib para
que el dashboard (Fase 4) los cargue sin tener que reentrenar en cada ejecución.

Ejecutar desde /src o /notebooks con el venv activado:
    python train_models.py
"""
import os

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from data_prep import CLIMA_COLS_CLUSTER, preparar_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    d = preparar_dataset()

    # --- Regresión: Random Forest con los hiperparámetros ganadores de la Fase 3 ---
    reg = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features=0.5, random_state=42, n_jobs=-1,
    )
    reg.fit(d.X_train_reg, d.y_reg_train)

    # --- Clasificación: Random Forest (ganador sobre KNN en la Fase 3) ---
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(d.X_train_clf, d.y_clf_train)

    # --- Clustering: K-Means (K=5, elegido por silueta en la Fase 3) + PCA para visualización ---
    scaler_cluster = StandardScaler()
    X_clima = scaler_cluster.fit_transform(d.df_limpio[CLIMA_COLS_CLUSTER])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clima)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_clima)

    # compress=3: el Random Forest de regresión (400 árboles) pesa >90MB sin comprimir,
    # por encima del límite de 100MB de GitHub y pesado para Streamlit Cloud
    joblib.dump(reg, os.path.join(MODELS_DIR, "modelo_regresion.joblib"), compress=3)
    joblib.dump(clf, os.path.join(MODELS_DIR, "modelo_clasificacion.joblib"), compress=3)
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "modelo_kmeans.joblib"), compress=3)
    joblib.dump(scaler_cluster, os.path.join(MODELS_DIR, "scaler_cluster.joblib"), compress=3)
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_cluster.joblib"), compress=3)
    joblib.dump(d.encoder_reg, os.path.join(MODELS_DIR, "encoder_reg.joblib"), compress=3)
    joblib.dump(d.scaler_reg, os.path.join(MODELS_DIR, "scaler_reg.joblib"), compress=3)

    print(f"Modelos guardados en {os.path.abspath(MODELS_DIR)}")
    print("Archivos:", sorted(os.listdir(MODELS_DIR)))


if __name__ == "__main__":
    main()
