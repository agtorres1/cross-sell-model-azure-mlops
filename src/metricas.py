"""Métricas de evaluación: AUC, KS, lift y tabla por deciles."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def calcular_auc(y_real, score):
    return roc_auc_score(y_real, score)


def calcular_ks(y_real, score):
    fpr, tpr, _ = roc_curve(y_real, score)
    return max(tpr - fpr)


def calcular_lift(y_real, score, top=0.10):
    tabla = pd.DataFrame({"y": np.asarray(y_real), "score": score})
    tabla = tabla.sort_values("score", ascending=False)
    n_top = int(len(tabla) * top)
    return tabla["y"].head(n_top).mean() / tabla["y"].mean()


def resumen(y_real, score):
    return {
        "auc": calcular_auc(y_real, score),
        "ks": calcular_ks(y_real, score),
        "lift_10": calcular_lift(y_real, score, 0.10),
        "lift_20": calcular_lift(y_real, score, 0.20),
    }


def asignar_decil(score):
    """Decil 1 = 10 % de mayor puntaje ... decil 10 = 10 % de menor puntaje."""
    orden = pd.Series(score).rank(method="first", ascending=False) - 1
    return (orden * 10 // len(orden) + 1).astype(int).values


def tabla_deciles(y_real, score):
    tabla = pd.DataFrame({"y": np.asarray(y_real), "score": score, "decil": asignar_decil(score)})
    d = tabla.groupby("decil").agg(clientes=("y", "size"), contrataron=("y", "sum"),
                                   puntaje_min=("score", "min"))
    d["tasa"] = d["contrataron"] / d["clientes"]
    d["lift"] = d["tasa"] / tabla["y"].mean()
    return d
