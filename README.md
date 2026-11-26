# Proyecto Integrador de Ciencia de Datos — Seoul Bike Sharing Demand

Universidad de La Salle — Facultad de Ingeniería / Maestría en Inteligencia Artificial
Asignatura: Ciencia de Datos — Docente: Fabián Camilo Castro Riveros
Autores: Harold Andrés Horta Suárez, Anderson Alexander González Ruiz

## Dataset

**Seoul Bike Sharing Demand** (UCI Machine Learning Repository, id 560, licencia CC BY 4.0).
8760 filas horarias (dic. 2017 – nov. 2018), 14 variables climáticas y temporales.
<https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand>

## Estructura del repositorio

```
.
├── data/
│   └── raw/SeoulBikeData.csv       # dataset original descargado de UCI
├── notebooks/
│   ├── 01_fase1_planteamiento_eda.ipynb
│   ├── 02_fase2_preprocesamiento_baseline.ipynb
│   └── 03_fase3_modelado_avanzado.ipynb
├── src/
│   ├── data_prep.py                 # pipeline de limpieza/features/partición (reutilizado por notebooks y dashboard)
│   └── train_models.py              # entrena y persiste los modelos finales para el dashboard
├── dashboard/
│   ├── app.py                       # dashboard interactivo (Streamlit)
│   └── models/                      # modelos serializados (joblib) — generados por train_models.py
├── requirements.txt
└── README.md
```

## Cómo reproducir

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ejecutar los notebooks en orden (01 -> 02 -> 03), o abrir en Jupyter:
jupyter lab notebooks/

# Entrenar y persistir los modelos finales para el dashboard:
cd src && python train_models.py && cd ..

# Levantar el dashboard interactivo:
cd dashboard && streamlit run app.py
```

## Resultados principales (Fase 3)

| Tarea | Modelo | Métrica | Resultado |
|---|---|---|---|
| Regresión (demanda horaria) | Random Forest + RandomizedSearchCV | R² (test) | **0.915** |
| Clasificación (estación del año) | Random Forest | Accuracy (val) | **0.849** |
| Clustering (perfiles climáticos) | K-Means (K=5) | Silueta / ANOVA | 0.288 / F=708.12, p≈0 |

Todos los objetivos específicos SMART del proyecto (R² ≥ 0.70, Accuracy ≥ 0.80, 3-5 clusters con p<0.05) se cumplieron.
