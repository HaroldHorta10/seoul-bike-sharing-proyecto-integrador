---
title: "Predicción y Segmentación de la Demanda de Bicicletas Compartidas en Seúl: un Enfoque Integral de Ciencia de Datos"
author:
  - "Harold Andrés Horta Suárez"
  - "Anderson Alexander González Ruiz"
date: "Noviembre de 2026"
---

# Resumen

Los sistemas de bicicletas compartidas dependen de un problema logístico crítico: distribuir la cantidad correcta de bicicletas en cada estación y momento del día. Este trabajo aborda el caso del sistema público de Seúl utilizando el dataset *Seoul Bike Sharing Demand* (UCI Machine Learning Repository, 8760 observaciones horarias, dic. 2017–nov. 2018). Se desarrolló un pipeline completo de ciencia de datos que combina tres tareas de modelado sobre el mismo conjunto de datos: regresión de la demanda horaria, clasificación de la estación del año a partir de condiciones observadas, y segmentación no supervisada de perfiles climáticos. El modelo final de regresión (Random Forest optimizado por validación cruzada) alcanzó R² = 0.915 sobre un conjunto de prueba independiente, superando el objetivo propuesto de 0.70. El clasificador de estaciones (Random Forest) alcanzó una exactitud de 0.849, y el análisis de clustering identificó 5 perfiles climáticos con diferencias de demanda estadísticamente significativas (ANOVA, p < 0.001). Durante el desarrollo se identificaron y corrigieron dos errores de fuga de datos, documentados explícitamente como parte del rigor metodológico del proyecto. Los resultados se integran en un dashboard interactivo que permite predicción de demanda en vivo y visualización de segmentos climáticos.

# 1. Introducción

## 1.1 Contexto y motivación

Los sistemas de *bike-sharing* urbano ofrecen una alternativa de bajo costo y bajas emisiones para trayectos cortos, pero su operación depende de un balance logístico fino: si faltan bicicletas en horas pico, el sistema pierde usuarios frente a otros medios de transporte; si sobran, la ciudad incurre en costos de mantenimiento y redistribución innecesarios. Seúl opera uno de los sistemas públicos de bicicletas más grandes de Asia. Este trabajo utiliza el registro horario de un año completo de operación para construir herramientas analíticas que apoyen decisiones operativas de distribución de flota.

## 1.2 Pregunta de investigación

**Principal:** ¿En qué medida las condiciones climáticas y temporales (hora del día, estación del año, día festivo) permiten predecir y explicar la demanda horaria de bicicletas compartidas en Seúl?

**Secundarias:**
1. ¿Qué variables climáticas presentan la asociación más fuerte con la demanda horaria?
2. ¿Es posible inferir la estación del año a partir del patrón climático y de demanda observado en una hora determinada?
3. ¿Existen perfiles horarios de condiciones climáticas (clústeres) con niveles de demanda claramente distintos entre sí, útiles para decisiones operativas de redistribución de flota?

## 1.3 Objetivos

**General:** Desarrollar un modelo analítico integral que permita predecir la demanda horaria de bicicletas compartidas en Seúl a partir de variables climáticas y temporales, como insumo para decisiones de distribución de flota.

**Específicos (SMART):**
1. Construir un modelo de regresión que explique al menos el 70% de la varianza (R²) de la demanda horaria, evaluado sobre un conjunto de prueba independiente.
2. Entrenar y comparar al menos dos algoritmos de clasificación capaces de predecir la estación del año a partir de las condiciones horarias, con exactitud superior al 80% sobre el conjunto de prueba.
3. Identificar, mediante al menos dos algoritmos de clustering, entre 3 y 5 perfiles horarios de condiciones climáticas con niveles de demanda significativamente distintos (ANOVA, p < 0.05).

# 2. Metodología

## 2.1 Dataset

Seoul Bike Sharing Demand (UCI, DOI 10.24432/C5F62R, licencia CC BY 4.0): 8760 filas horarias, 14 variables (`Rented_Bike_Count`, `Hour`, `Temperature`, `Humidity`, `Wind_speed`, `Visibility`, `Dew_point_temp`, `Solar_Radiation`, `Rainfall`, `Snowfall`, `Seasons`, `Holiday`, `Functioning_Day`, `Date`). 0% de valores faltantes certificado por UCI y verificado de forma independiente.

## 2.2 Análisis exploratorio

El análisis univariado reveló asimetría moderada en la demanda (skew = 1.15) y asimetría extrema en `Rainfall`/`Snowfall` (variables de evento raro, con la mayoría de observaciones en cero). El análisis bivariado identificó correlación máxima entre demanda y `Temperature` (r = 0.54) y `Hour` (r = 0.41), y multicolinealidad relevante entre `Temperature` y `Dew_point_temp` (r = 0.91). El diagnóstico de calidad detectó una incoherencia de dominio: 295 horas (3.4%) con `Functioning_Day = "No"` presentan demanda forzada a cero de forma artificial, no representativa de demanda real.

## 2.3 Preprocesamiento

Se excluyeron las 295 horas sin servicio. Se aplicó transformación `log1p` a `Rented_Bike_Count`, `Rainfall` y `Snowfall` para estabilizar la asimetría (1.15 → -0.77 en la variable objetivo). Se crearon tres variables derivadas con respaldo de dominio: `Mes` (estacionalidad fina), `Es_Fin_de_Semana` (patrones de movilidad) y `Franja_Horaria` (bloques de uso horario). Los datos se particionaron 70/15/15 (train/val/test) de forma estratificada por `Seasons`.

**Nota metodológica — dos conjuntos de features independientes.** Dado que el mismo dataset alimenta tanto el problema de regresión como el de clasificación, se construyeron pipelines de codificación/escalamiento separados para cada tarea. Esto fue necesario tras detectar dos errores de fuga de datos durante el desarrollo: (1) un primer diseño incluía `Seasons` simultáneamente como feature y como variable objetivo de clasificación; (2) tras corregir lo anterior, se detectó que `Mes` determina a `Seasons` de forma determinística (cada mes mapea a una única estación en este dataset), por lo que su inclusión en el conjunto de features de clasificación constituía la misma fuga de datos por una vía indirecta. Ambos errores fueron identificados mediante inspección de resultados anómalamente perfectos (accuracy = 100%) y corregidos antes de reportar resultados finales. Todo ajuste de codificadores y escaladores (`fit`) se realizó exclusivamente sobre el conjunto de entrenamiento.

## 2.4 Modelos

**Regresión:** modelos lineales regularizados (Ridge, Lasso, ElasticNet) y Random Forest Regressor, con optimización de hiperparámetros mediante `RandomizedSearchCV` (validación cruzada de 5 particiones, 20 combinaciones muestreadas).

**Clasificación:** K-Nearest Neighbors y Random Forest Classifier, elegidos por representar paradigmas opuestos (basado en distancia vs. basado en ensamble de reglas).

**Clustering:** K-Means (número óptimo de clusters determinado por coeficiente de silueta en el rango permitido 3-5) y clustering jerárquico aglomerativo (enlace de Ward) como segundo algoritmo, sobre las 8 variables climáticas estandarizadas.

**Perceptrón multicapa (MLP):** arquitectura feedforward (21→64→32→1) implementada en PyTorch, optimizador Adam, función de pérdida MSE, 150 épocas, para el problema de regresión.

**Interpretabilidad:** SHAP (`TreeExplainer`) sobre el modelo Random Forest final, para cuantificar la contribución marginal de cada variable a las predicciones individuales.

# 3. Resultados

| Modelo | Tarea | Métrica principal | Resultado |
|---|---|---|---|
| Ridge / Lasso / ElasticNet | Regresión | R² (validación) | 0.653–0.655 |
| Random Forest (hiperparámetros por defecto) | Regresión | R² (validación) | 0.876 |
| **Random Forest + RandomizedSearchCV** | **Regresión** | **R² (test)** | **0.915** |
| MLP (PyTorch) | Regresión | R² (validación) | 0.476 |
| KNN | Clasificación | Accuracy (validación) | 0.763 |
| **Random Forest** | **Clasificación** | **Accuracy (validación)** | **0.849** |
| K-Means (K=5) | Clustering | Silueta | 0.288 |
| K-Means vs. Jerárquico | Clustering | Adjusted Rand Index | 0.366 |
| ANOVA demanda ~ cluster | Clustering | F / p-valor | 708.12 / p < 0.001 |

El baseline de regresión (Temperature → demanda, modelo lineal simple) alcanzó R² = 0.304; el baseline de clasificación (clase más frecuente) alcanzó accuracy = 0.261. Todos los modelos finales superan ampliamente estos pisos de referencia.

El análisis SHAP identificó como variables de mayor impacto, en orden: `Temperature`, `Hour`, `Humidity`, `Seasons_Winter` y `log_Rainfall` — coincidentes con las correlaciones de mayor magnitud detectadas en el análisis exploratorio inicial, lo que confirma consistencia metodológica entre las fases del proyecto.

Los cinco perfiles climáticos identificados por K-Means muestran demanda promedio que va de 1229 bicicletas/hora (cluster cálido y seco, alta radiación solar) a 107 bicicletas/hora (cluster de alta humedad con nevadas). La precipitación y la nieve emergen como los factores con mayor efecto reductor sobre la demanda, por encima incluso de la temperatura baja por sí sola.

# 4. Discusión

Los tres objetivos específicos SMART del proyecto fueron alcanzados. El margen con el que Random Forest superó tanto a los modelos lineales regularizados como al MLP es consistente con hallazgos reportados en la literatura sobre datos tabulares de tamaño pequeño-mediano, donde los modelos de ensamble basados en árboles suelen igualar o superar a redes neuronales simples; estas últimas requieren típicamente mayor volumen de datos o arquitecturas más sofisticadas (regularización, *embeddings* para variables categóricas) para ser competitivas en este tipo de problema.

El hallazgo más relevante desde el punto de vista metodológico no fue un resultado de modelado sino los dos errores de fuga de datos detectados durante el desarrollo. Ambos casos ilustran un principio general: verificar que el target de una tarea no esté presente como feature no es suficiente cuando existen variables derivadas que son proxies determinísticos del target por construcción (como `Mes` respecto de `Seasons`). Resultados sospechosamente perfectos (accuracy = 100%) fueron la señal que motivó la investigación y corrección, antes de reportar cualquier métrica final.

Desde la perspectiva de negocio, los perfiles de clustering ofrecen un insumo directamente accionable: los segmentos de alta humedad/precipitación son candidatos naturales para reducir la flota disponible, mientras que los segmentos cálidos y secos justifican refuerzo de stock — una guía más granular que la simple estacionalidad calendario.

# 5. Conclusiones

Se desarrolló un pipeline reproducible de ciencia de datos que resuelve simultáneamente un problema de regresión, uno de clasificación y uno de clustering sobre el mismo dataset público, cumpliendo los tres objetivos específicos propuestos. El modelo final de regresión (R² = 0.915) y de clasificación (accuracy = 0.849) superan ampliamente sus respectivos pisos de referencia. El proceso de desarrollo evidenció la importancia de la verificación activa de fuga de datos como práctica metodológica central, no como un chequeo superficial de "¿está el target en las features?" sino como una pregunta más amplia sobre proxies determinísticos.

# 6. Trabajo futuro

- Incorporar variables de ubicación geográfica (si estuvieran disponibles por estación de bicicletas, no solo a nivel ciudad) para pasar de un modelo de demanda agregada a un modelo de redistribución por estación individual.
- Explorar arquitecturas de MLP más profundas con regularización (dropout, batch normalization) y comparar contra gradient boosting (XGBoost/LightGBM) como extensión del bloque de ensambles.
- Incorporar validación temporal (*time series split*) en lugar de partición aleatoria estratificada, dado que los datos tienen estructura temporal secuencial que un split aleatorio no respeta completamente.
- Desplegar el dashboard en un servicio de hosting público (Streamlit Community Cloud u otro) para habilitar acceso remoto fuera del entorno de demostración local.
