# Cross-Sell Propensity Model · Azure MLOps

Modelo supervisado que predice qué clientes bancarios tienen mayor probabilidad de **adquirir un paquete de productos**, para priorizar campañas comerciales. Desarrollado con metodología KDD y diseñado para llevarse a producción end-to-end con **Azure Machine Learning**.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2E8B57)
![Azure ML](https://img.shields.io/badge/Azure_ML-SDK_v2-0078D4?logo=microsoftazure&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Estado](https://img.shields.io/badge/estado-modelado_en_curso-yellow)

---

## El problema

El análisis exploratorio mostró que los clientes con `Target=1` tienen **más productos activos, más transacciones y actividad más reciente** que los de `Target=0`. Es el patrón opuesto a una fuga: el target no mide churn, mide **propensión de compra y engagement**.

Ese hallazgo definió todo lo demás:

- **La ABT** se enfoca en adopción, transaccionalidad, balances y uso de canales.
- **El uso de negocio** es priorizar colas de contacto comercial (call center, campañas dirigidas, notificaciones en la app), no retener.
- **La evaluación** se hace con métricas de campaña (lift y ganancia por decil), no solo con AUC.

---

## Enfoque en 30 segundos

| | |
|---|---|
| **Metodología** | KDD |
| **Unidad de análisis** | Cliente (`client_id`), con corte al mes 6 |
| **Ventanas** | 6 meses de historia · 1 mes de lead · 2 meses de predicción |
| **Target** | `Y=1` si el cliente adquiere el paquete en el mes 8 o 9 |
| **Modelos** | Regresión logística (baseline), árbol CART, Random Forest, LightGBM |
| **Tuning** | `RandomizedSearchCV` → `GridSearchCV` sobre el mejor modelo |
| **Métricas** | Lift y ganancia acumulada (top 10/20/30%), KS, AUC-ROC, PSI |
| **Despliegue** | Endpoint de Azure ML para scoring de campañas |

```
mes:  1    2    3    4    5    6  |  7  |  8    9
     |——— historia (features) ———|lead | predicción
                                 ▲
                            corte de la ABT
```

- **Historia (meses 1–6):** de acá salen todas las features.
- **Lead (mes 7):** tiempo real para scorear, armar listas y contactar.
- **Predicción (meses 8–9):** ventana del target.

**Universo:** clientes con al menos 9 meses continuos de historia, excluyendo a quienes ya tienen el paquete al mes 6.

---

## Estado

| Etapa | Notebook | Estado |
|---|---|---|
| EDA y definición del universo | `01_eda_y_definicion_universo` | ✅ |
| Construcción de la ABT | `02_construccion_abt` | ✅ |
| Entrenamiento de modelos | `03_entrenamiento_modelos` | 🟡 En curso |
| Evaluación con métricas de negocio | `04_evaluacion_metricas_negocio` | ⏳ |
| Pipeline, endpoint y monitoreo en Azure ML | `azure/` | ⏳ Diseñado |

Los resultados (AUC, KS, lift del primer decil) se publican acá al cerrar el modelado.

---

## ABT

Una fila por `client_id` al corte del mes 6.

- **Identity features:** edad, antigüedad, ingreso declarado.
- **Transform features:** rangos etarios, total de productos activos, flags de tenencia por producto.
- **Aggregate features:** para cada variable transaccional o de balance, **8 estadísticos × 2 horizontes** (últimos 3 y últimos 6 meses) = 16 features:
  mínimo, máximo, media, mediana, suma, cantidad de meses con actividad, dispersión y variación relativa entre el primer y el último mes.

Los dos horizontes permiten distinguir el comportamiento reciente del comportamiento estructural del cliente.

---

## Decisiones de diseño

- **Nulos con criterio bancario:** un monto o conteo vacío significa "sin actividad" y se imputa con 0. En el resto se usa mediana o media. Las categóricas nulas van a una categoría explícita (`DESCONOCIDO`).
- **Outliers:** se acotan (capping p1–p99) en lugar de eliminarse; en banca los extremos suelen ser clientes reales de alto valor.
- **Encoding:** target encoding para categóricas de alta cardinalidad, one-hot para las de baja.
- **Desbalance:** `class_weight='balanced'`, undersampling controlado y ajuste del threshold según el punto de corte comercial.
- **Validación:** split estratificado train/validation/test con Stratified K-Fold.
- **Overfitting:** tabla de deciles con chequeo de orden monotónico y consistencia train/test.
- **Estabilidad:** PSI y criterios explícitos de recalibración (caída de AUC/KS, pérdida de importancia de variables, cambios en la política comercial).

---

## Productivización con Azure MLOps

```
Datos raw ─► Datastore + Data Asset versionado
                    │
                    ▼
            Azure ML Pipeline (SDK v2 / YAML)
            ├─ preprocesamiento / ABT    ← src/features.py
            ├─ entrenamiento + tuning    ← src/train.py
            └─ evaluación + gate de calidad
                    │  registra solo si supera al modelo en producción en AUC y KS
                    ▼
            Model Registry (MLflow) ─► Endpoint ─► scoring de campañas
                                          │
                                          ▼
                     Monitoreo de data drift y performance
                     → alertas y disparo de reentrenamiento
```

El mismo código de `src/` corre en los notebooks, en el pipeline y en el endpoint: no hay lógica duplicada y se evita el *training-serving skew*.

---

## Estructura

```
data/        datasets raw, intermedios y ABT
notebooks/   las 4 etapas de KDD
src/         limpieza, features, entrenamiento y scoring reutilizables
azure/       componentes de Azure ML: pipelines, endpoints, monitoreo
models/      artefactos (.pkl / MLflow) y referencia al Model Registry
docs/        diccionario de datos, reportes y decisiones de arquitectura
```

---

## Stack

Python · pandas · scikit-learn · LightGBM · MLflow · Azure Machine Learning (SDK v2: Data Assets, Pipelines, Model Registry, Endpoints, Monitoring)

---

**Autora:** Agustina Torres · Ingeniería en Sistemas, UTN FRBA
