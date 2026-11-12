# Decisiones del modelado

Registro de las decisiones tomadas y su justificación.

## 1. Validación: train/test + cross-validation

- Se separa un 20 % de test estratificado que se usa **una sola vez**, al final.
- Todo lo demás (comparar modelos, ajustar hiperparámetros) se hace con CV estratificado de 5 folds sobre train, con los **mismos folds** para todos los modelos.
- No se usa un set de validación aparte: con ~23.000 clientes, el k-fold cumple ese rol sin achicar el train.

## 2. Métricas: AUC, KS y lift

- El AUC mide el orden de toda la lista; el lift del top 10–20 % mide lo que le importa a Marketing (acertar arriba de la lista).
- La búsqueda de hiperparámetros optimiza AUC por practicidad (sklearn lo trae listo); el ganador se valida después con KS y lift.

## 3. Baseline: regresión logística

- Modelo simple y explicable como piso (AUC CV 0,820).
- Con `StandardScaler` dentro de un `Pipeline`, para que el escalado se aprenda solo con los datos de cada fold.

## 4. Modelo elegido: LightGBM

- Le gana a la logística en las cuatro métricas (AUC 0,833 vs. 0,820; lift 10 % 2,55 vs. 2,44). La diferencia es chica pero consistente, y en lift del top 10 % equivale a ~75 clientes interesados más en una campaña de ~2.300.
- Contra: es menos explicable. Se compensa con SHAP.

## 5. Hiperparámetros: Randomized Search y luego simplificación

- Randomized Search (30 combinaciones) en lugar de Grid Search: con el mismo presupuesto prueba muchos más valores distintos de cada perilla.
- El ajuste **no mejoró el AUC** respecto de los valores por defecto (0,834 en ambos), pero dio árboles más prudentes: 16 hojas, mínimo 148 clientes por hoja, subsample y colsample < 1.
- El ganador quedó cerca del borde inferior del rango (learning rate 0,011, 16 hojas). Se probó una grilla de árboles × learning rate y **más árboles no mejoran**: con 100 árboles y lr 0,05 el AUC es 0,833 vs. 0,834 con 502 árboles, diferencia menor al ruido entre folds (~0,003).
- **Decisión: 100 árboles, learning rate 0,05.** A igual rendimiento, se elige el modelo más simple: entrena y puntúa más rápido y es más fácil de defender.

Perillas finales: `n_estimators=100`, `learning_rate=0.05`, `num_leaves=16`, `min_child_samples=148`, `subsample=0.81`, `subsample_freq=1`, `colsample_bytree=0.76`.

## 6. Sin balanceo de clases

- La tasa de positivos (29 %) no es extrema y las métricas usadas (AUC, KS, lift) dependen solo del **orden** de los clientes, que el balanceo no mejora.
- Ni undersampling ni class weights: el undersampling descarta datos y ambos distorsionan las probabilidades (habría que recalibrarlas).

## 7. Corte de campaña: decil 7

- Con costo por llamada de $10.000 y ganancia por venta de $100.000, conviene llamar a un decil mientras su tasa de contratación supere 10 %.
- El decil 7 (14 %) es el último que lo supera. Cortar ahí da la ganancia máxima en test (~$100,8 M vs. ~$90,1 M llamando a todos).
- Si la capacidad de Marketing es menor, se corta antes, empezando por los primeros deciles.
- Supuesto: toda contratación se atribuye a la llamada. El efecto real (cuántos habrían comprado igual) solo se mide con un A/B test.

## Limitaciones conocidas

1. **Selección de variables con todos los clientes.** Las 50 variables se eligieron con la importancia de LightGBM sobre los 23.191 clientes, incluido el test. El AUC de test puede ser levemente optimista.
2. **Target encoding con todos los clientes.** `Tarj_pct_target` y `Region_pct_target` se calcularon en `03_abt` con todos los clientes. Recalculadas solo con train, el resultado en test es **idéntico** (AUC 0,8395, lift 10 % 2,62): las tasas casi no cambian y los árboles solo usan el orden. En `src/` el encoding debe aprenderse solo con train y guardarse para producción.
3. **Sin validación out-of-time.** Train y test son del mismo `t0`. Una validación con un `t0` posterior anticiparía mejor el rendimiento en producción.
4. **Concentración en una variable.** `Tarj_pct_target` pesa ~5 veces más que la segunda: si cambia la oferta de tarjetas del banco, el modelo se ve muy afectado. Monitorear su distribución (PSI) en producción.
