# Predicción de propensión a nuevo producto (cross-sell) con Azure MLOps

Modelo supervisado de Machine Learning para predecir qué clientes bancarios tienen mayor probabilidad de contratar un nuevo paquete de productos (cross-selling), desarrollado bajo la metodología **KDD** (Knowledge Discovery in Databases) y diseñado para su productivización end-to-end con **Azure Machine Learning**.

---

## 📌 Nota sobre la Definición del Target

El análisis exploratorio inicial evidenció que los clientes con `Target = 1` presentan mayor cantidad de productos activos, mayor volumen transaccional y mayor actividad reciente en comparación con los clientes con `Target = 0`. Este comportamiento descarta la hipótesis de fuga (*churn*) y valida que el target mide **propensión de compra / engagement**, no abandono.

Esta validación temprana condicionó directamente:

1. **El diseño de la ABT (Parte 1):** Foco en variables de adopción, transaccionalidad, saldos y uso de canales.
2. **El enfoque de despliegue (Parte 2):** Priorización de colas de contacto comercial (Call Center, campañas dirigidas y notificaciones en Mobile App) y no retención.

---

## 📂 Estructura del Repositorio

```text
├── data/           # Datasets raw, intermedios y tablas analíticas (ABT)
├── notebooks/      # Flujo modular de Experimentación y Modelado (KDD - 4 etapas)
│   ├── 01_eda_y_definicion_universo.ipynb
│   ├── 02_construccion_abt.ipynb
│   ├── 03_entrenamiento_modelos.ipynb
│   └── 04_evaluacion_metricas_negocio.ipynb
├── src/            # Funciones productivas reutilizables (limpieza, features, scoring)
├── azure/          # Componentes de Azure ML (pipelines SDK v2, endpoints, monitoreo)
├── models/         # Model Registry y serialización (.pkl / MLflow)
└── docs/           # Diccionario de datos, reportes y decisiones de arquitectura
```

> **Principio de diseño:** lo explorado y validado en `notebooks/` se traslada como código reutilizable a `src/`, y ese mismo código es el que ejecutan los componentes del pipeline en `azure/`. No hay lógica duplicada entre experimentación y productivización.

---

## 🔬 Parte 1 — Modelado (Metodología KDD)

Aplicada sobre el dataset histórico de clientes bancarios (`client_id`, transacciones, saldos de cuentas, tenencia de productos y uso de canales), siguiendo las fases clásicas de KDD:

### 1. Objetivo y Definición del Universo (Selection & Business Understanding)

**Objetivo de Negocio:** estimar la probabilidad de que un cliente adquiera un paquete bancario para optimizar la tasa de conversión en campañas de marketing y colas de atención.

**Ventana Temporal (9 meses en total):**

- **Training / Historical Window (6 meses):** período de observación histórica para generar variables explicativas.
- **Lead Window (1 mes):** brecha operativa requerida por el negocio (ejecución del scoring, armado de listas y contacto) para evitar data leakage.
- **Prediction Window (2 meses):** ventana de desempeño donde se observa si ocurre la compra.

**Filtros del Universo:**

- Clientes con al menos 9 meses continuos de historial en la base.
- Exclusión de clientes que ya posean el paquete activo en o antes del mes 6 (punto de corte).

**Definición del Target (Y):**

```
Y = 1  si el cliente adquiere el paquete en la ventana de predicción (meses 8 o 9)
Y = 0  en caso contrario
```

**Estrategia ante Desbalance:** evaluación de balanceo mediante `class_weight='balanced'`, técnicas de undersampling controlado y calibración del umbral de decisión (*threshold tuning*) según el punto de corte óptimo de negocio.

### 2. Limpieza y Calidad de Datos (Preprocessing)

**Valores Faltantes (Missing Values):**

- *Numéricos:* imputación con 0 cuando refleja ausencia de transacciones/saldo; imputación con la mediana (o media) cuando el cero distorsiona la distribución.
- *Categóricos:* imputación mediante categoría explícita `'NO DATA'` / `'DESCONOCIDO'`.

**Outliers:** detección y tratamiento mediante rango intercuartílico (IQR / Boxplot) y técnica de corte a 3σ (o percentiles 1% y 99%).

**Codificación:** Target Encoding para variables de alta cardinalidad y One-Hot Encoding para categorías nominales de baja cardinalidad.

### 3. Construcción de la ABT (Transformation & Feature Engineering)

Consolidación de la Analytic Base Table a nivel `client_id` al corte del mes 6:

- **Identity Features:** atributos directos del cliente (edad, antigüedad, ingresos declarados).
- **Transform Features:** variables derivadas con lógica de negocio (tramos etarios, cantidad total de productos activos, flags binarios de tenencia).
- **Aggregate Features (Metodología de las 16 features por variable numérica):**

  Para cada métrica transaccional y de saldo (caja de ahorro, movimientos, etc.), cálculo de 8 estadísticos en 2 horizontes temporales (últimos 3 meses y últimos 6 meses):

  - Mínimo (`min`)
  - Máximo (`max`)
  - Promedio (`mean`)
  - Mediana (`median`)
  - Suma (`sum`)
  - Conteo de actividad distinta de cero (`non_cero`)
  - Varianza / Dispersión (`var`)
  - Variación relativa entre primer y último mes del período

### 4. Selección de Variables

- Eliminación de features con varianza cero o valores constantes.
- Análisis de multicolinealidad vía matriz de correlación.
- Reducción de dimensionalidad (PCA con estandarización previa vía `StandardScaler`) y ranking de importancia de variables mediante modelos basados en árboles (Random Forest / LightGBM) como primer filtro de selección.

### 5. Algoritmos y Entrenamiento (Data Mining)

**Partición de Datos:** split estratificado Train / Validation / Test mediante Stratified K-Fold Cross Validation para preservar la proporción del target.

**Modelos Evaluados:**

- Regresión Logística (baseline lineal e interpretable)
- Árbol de Decisión (CART)
- Random Forest Classifier (ensamble por bagging)
- LightGBM (Gradient Boosting optimizado para datos tabulares y alta performance)

**Optimización de Hiperparámetros:** búsqueda combinada con `RandomizedSearchCV` (exploración amplia del espacio) y `GridSearchCV` (afinado fino del mejor modelo).

### 6. Evaluación de Negocio e Interpretación (Evaluation)

**Métricas Técnicas:** curva ROC, AUC-ROC y matriz de confusión ajustada al umbral de contacto óptimo.

**Métricas de Negocio y Ranking:**

- **Curva Lift y Ganancia Acumulada:** evaluación del factor de enriquecimiento en los primeros deciles (top 10%, 20%, 30%).
- **Estadístico KS (Kolmogorov-Smirnov):** verificación de la máxima separación entre compradores y no compradores.
- **Tabla de Deciles:** validación del ordenamiento decreciente estricto y consistencia de performance entre Train y Test para descartar sobreajuste (overfitting).

**Estabilidad y Recalibración:** chequeo de estabilidad poblacional (PSI) y establecimiento de criterios de recalibración (caída de AUC/KS en nuevos meses, pérdida de importancia de variables, cambios en políticas comerciales).

**📓 Notebooks de trabajo:**

- [`notebooks/01_eda_y_definicion_universo.ipynb`](notebooks/01_eda_y_definicion_universo.ipynb)
- [`notebooks/02_construccion_abt.ipynb`](notebooks/02_construccion_abt.ipynb)
- [`notebooks/03_entrenamiento_modelos.ipynb`](notebooks/03_entrenamiento_modelos.ipynb)
- [`notebooks/04_evaluacion_metricas_negocio.ipynb`](notebooks/04_evaluacion_metricas_negocio.ipynb)

---

## 🚀 Parte 2 — Productivización con Azure MLOps

```
data/ (raw)
   │
   ▼
Azure ML Datastore + Data Asset  ──────────────►  Versionado y linaje del dataset
   │
   ▼
Azure ML Pipeline (Entrenamiento)
   ├─ Componente: Preprocesamiento / ABT   (reutiliza src/features.py)
   ├─ Componente: Entrenamiento + Tuning   (reutiliza src/train.py)
   └─ Componente: Evaluación y Quality Gate
   │
   ▼
Model Registry (Azure ML)  ─────────────────────►  Artefacto, métricas y parámetros (MLflow)
   │
   ▼
Managed Online / Batch Endpoint ────────────────►  Scoring para campañas de cross-sell
   │
   ▼
Monitoreo MLOps (Data Drift & Performance) ─────►  Alertas y trigger de reentrenamiento
```

- **Registro de Dataset y Tracking:** carga del dataset como Data Asset versionado en Azure ML Datastore y seguimiento de corridas con MLflow integrado (métricas, parámetros y artefactos).
- **Pipeline Declarativo:** construcción mediante Azure ML SDK v2 / YAML vinculando los componentes de `src/` sin duplicar código.
- **Quality Gate:** registro condicional en el Model Registry solo si el modelo candidato supera el umbral de AUC y KS del modelo en producción.
- **Endpoint de Inferencia:** Managed Online / Batch Endpoint para scoring periódico sobre la cartera de clientes, aplicando las mismas transformaciones para evitar *training-serving skew*.
- **Monitoreo Continuo:** detección de data drift en variables de entrada y degradación de performance en el tiempo para disparar reentrenamientos automatizados.

---

## 📊 Resultados

| Métrica | Valor |
|---|---|
| AUC-ROC Final | *(a completar con corrida final)* |
| Estadístico KS | *(a completar con corrida final)* |
| Lift en 1er Decil | *(a completar con corrida final)* |

**Recomendación Comercial:** focalización de la campaña en los deciles de mayor propensión para maximizar el ROI del canal de contacto.
