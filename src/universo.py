"""Universo (a quién se le aplica el modelo) y target."""
import pandas as pd

from .config import MESES_OBSERVACION, MESES_LEAD, MESES_RESPUESTA
from .data import mes, meses


def construir_universo(df, t0, con_target=True):
    """Clientes elegibles en t0.

    Reglas (01_universo_y_target):
      - historia completa: el cliente aparece todos los meses de la ventana;
      - en t0 no tiene el paquete (Package_Active = 'No');
      - en t0 no tiene tarjeta co-branding (CreditCard_CoBranding = 'No').

    con_target=True (entrenamiento): la ventana va de t0-5 a t0+3, porque
    hace falta ver la respuesta. TGT = 1 si tiene Target = 1 en t0+2 o t0+3.

    con_target=False (puntuación): solo se exige la ventana de observación
    (t0-5 a t0). No hay TGT: es lo que se quiere predecir.
    """
    desde = -(MESES_OBSERVACION - 1)
    hasta = MESES_LEAD + MESES_RESPUESTA if con_target else 0
    ventana = meses(t0, desde, hasta)

    presentes = df[df["Month"].isin(ventana)].groupby("client_id")["Month"].nunique()
    historia_completa = presentes[presentes == len(ventana)].index

    foto = df[df["Month"] == mes(t0)]
    elegibles = foto[(foto["Package_Active"] == "No")
                     & (foto["CreditCard_CoBranding"] == "No")]["client_id"]

    universo = pd.DataFrame({"client_id": sorted(set(historia_completa) & set(elegibles))})

    if con_target:
        respuesta = meses(t0, MESES_LEAD + 1, MESES_LEAD + MESES_RESPUESTA)
        positivos = df[df["Month"].isin(respuesta) & (df["Target"] == 1)]["client_id"].unique()
        universo["TGT"] = universo["client_id"].isin(positivos).astype(int)

    return universo
