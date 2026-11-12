# Métricas del modelo

Modelo de propensión a cross-sell. Universo: 23.191 clientes, tasa de `TGT = 1` de 29,42 %.
Todas las métricas se calcularon con las funciones `calcular_auc`, `calcular_ks` y `calcular_lift` del notebook de modelado.

## Partición

| Conjunto | Clientes | Tasa TGT = 1 | Uso |
|---|---|---|---|
| Train | 18.552 | 29,42 % | Entrenar y comparar modelos (CV estratificado de 5 folds) |
| Test | 4.639 | 29,42 % | Evaluación final, usado una sola vez |

Split estratificado 80/20, `random_state=42`. Techo teórico del lift: 1 / 0,294 ≈ 3,4.

## Comparación de modelos (CV sobre train)

| Modelo | AUC | KS | Lift 10 % | Lift 20 % |
|---|---|---|---|---|
| Regresión logística (baseline) | 0,820 | 0,495 | 2,44 | 2,25 |
| LightGBM, valores por defecto | 0,834 | 0,509 | 2,48 | 2,28 |
| LightGBM, ganador del Randomized Search (502 árboles, lr 0,011) | 0,834 | 0,501 | 2,58 | 2,30 |
| **LightGBM final (100 árboles, lr 0,05)** | **0,833** | **0,500** | **2,55** | **2,29** |

Variación del AUC entre folds: desvío ≈ 0,003. Diferencias menores a eso se consideran ruido.

## Búsqueda de hiperparámetros

Randomized Search, 30 combinaciones × 5 folds, criterio AUC.

- Mejor combinación: AUC CV 0,834 (train 0,868).
- Peores combinaciones (learning rate ≈ 0,15–0,19): AUC CV ≈ 0,80 con AUC train ≈ 1,00 → overfitting.

Cantidad de árboles vs. AUC en CV (resto de perillas fijas):

| Learning rate | 100 árboles | 200 | 300 | 500 |
|---|---|---|---|---|
| 0,011 | 0,827 | 0,831 | 0,832 | 0,834 |
| 0,03 | 0,832 | 0,834 | 0,833 | 0,832 |
| 0,05 | **0,833** | 0,833 | 0,831 | 0,828 |
| 0,1 | 0,832 | 0,829 | 0,825 | 0,819 |

## Modelo final en test

| Métrica | Train | Test |
|---|---|---|
| AUC | 0,860 | **0,840** |
| KS | 0,551 | **0,506** |
| Lift 10 % | 2,78 | **2,62** |
| Lift 20 % | 2,45 | **2,37** |

Test en línea con el CV (0,833) y brecha train–test chica: sin overfitting.

## Tabla por deciles (test)

| Decil | Clientes | Contrataron | Tasa | Lift | Puntaje mínimo |
|---|---|---|---|---|---|
| 1 | 464 | 357 | 76,9 % | 2,62 | 0,664 |
| 2 | 464 | 291 | 62,7 % | 2,13 | 0,523 |
| 3 | 464 | 222 | 47,8 % | 1,63 | 0,409 |
| 4 | 464 | 153 | 33,0 % | 1,12 | 0,326 |
| 5 | 464 | 138 | 29,7 % | 1,01 | 0,255 |
| 6 | 464 | 106 | 22,8 % | 0,78 | 0,189 |
| 7 | 464 | 66 | 14,2 % | 0,48 | 0,083 |
| 8 | 464 | 25 | 5,4 % | 0,18 | 0,029 |
| 9 | 464 | 6 | 1,3 % | 0,04 | 0,014 |
| 10 | 463 | 1 | 0,2 % | 0,01 | 0,012 |

Deciles del mismo tamaño y tasa siempre descendente.

## Corte de campaña (test)

Costo por llamada $10.000, ganancia por venta $100.000 → equilibrio en 10 % de contratación.

| Llamo a deciles | Llamados | Compran | Acierto | Captura | Ganancia |
|---|---|---|---|---|---|
| 1 | 464 | 357 | 77 % | 26 % | $31,1 M |
| 1–3 | 1.392 | 870 | 62 % | 64 % | $73,1 M |
| 1–5 | 2.320 | 1.161 | 50 % | 85 % | $92,9 M |
| **1–7** | **3.248** | **1.333** | **41 %** | **98 %** | **$100,8 M** |
| 1–8 | 3.712 | 1.358 | 37 % | 99 % | $98,7 M |
| Todos | 4.639 | 1.365 | 29 % | 100 % | $90,1 M |

## Variables más importantes (SHAP sobre test)

| # | Variable | Efecto |
|---|---|---|
| 1 | `Tarj_pct_target` (tipo de tarjeta) | Domina: ~5 veces la segunda |
| 2 | `SavingAccount_Credits_Amounts_mean` | Más ingresos a la caja de ahorro → sube |
| 3 | `Age_grp_ord` | Mayor edad → sube |
| 4 | `SavingAccount_Days_with_use_mean` | Más días de uso → sube |
| 5 | `SavingAccount_Transfer_In_Amount_mean` | Muchas transferencias recibidas → **baja** |
| 6 | `CreditCard_Total_Spending_max_u3m` | Más gasto con tarjeta (3 meses) → sube |

Las 10 primeras variables concentran ~73 % de la importancia.
