"""Limpieza de la ventana de observación (02_limpieza).

Funciones puras: no aprenden nada de los datos. Los números que sí se
aprenden (límites del capping) se calculan en features.PreparadorABT.fit
y se reciben acá como parámetro.
"""
import numpy as np
import pandas as pd

from .config import MESES_OBSERVACION
from .data import mes, meses


def ventana_observacion(df, clientes, t0):
    """Filas mensuales de los clientes indicados, de t0-5 a t0."""
    ventana = meses(t0, -(MESES_OBSERVACION - 1), 0)
    filas = df[df["Month"].isin(ventana) & df["client_id"].isin(clientes)]
    return filas.copy()


def imputar_saldo_promedio(dv):
    """Saldo promedio faltante = saldo inicial + créditos - débitos del mes."""
    estimado = (dv["SavingAccount_Balance_FirstDate"]
                + dv["SavingAccount_Credits_Amounts"]
                - dv["SavingAccount_Debits_Amounts"])
    dv["SavingAccount_Balance_Average"] = dv["SavingAccount_Balance_Average"].fillna(estimado)
    return dv


def aplicar_capping(dv, limites):
    """Recorta cada variable en su límite (p95 de los valores positivos en train)."""
    for col, limite in limites.items():
        dv[col] = dv[col].clip(upper=limite)
    return dv


def atributo_mas_cercano(df, columna, t0):
    """Valor de `columna` para cada cliente, tomado del mes más cercano a t0
    en el que esté informado.

    Region y CreditCard_Product solo se registran desde marzo 2019. Son
    atributos estables, así que se toma el dato más cercano a la foto:
    con t0 = enero 2019 sale de marzo; con t0 = abril 2019, de abril.
    Ante un empate de distancia se prefiere el mes anterior a t0.
    """
    informados = df.loc[df[columna].notna(), ["client_id", "Month", columna]].copy()
    distancia = (pd.PeriodIndex(informados["Month"], freq="M")
                 - pd.Period(t0, freq="M")).map(lambda d: d.n)
    informados["dist"] = np.abs(distancia)
    informados["futuro"] = (distancia > 0).astype(int)
    informados = informados.sort_values(["client_id", "dist", "futuro"])
    return informados.drop_duplicates("client_id").set_index("client_id")[columna]


def tipo_de_tarjeta(df, foto, t0):
    """Tipo de tarjeta por cliente: 'No tiene' si no tiene tarjeta activa en t0.

    Si tiene tarjeta pero no hay tipo informado queda NaN; se completa con la
    moda aprendida en train (features.PreparadorABT).
    """
    producto = foto["client_id"].map(atributo_mas_cercano(df, "CreditCard_Product", t0))
    tiene = foto["CreditCard_Active"] == "Yes"
    return pd.Series(np.where(tiene, producto, "No tiene"), index=foto["client_id"].values)


def limpiar_ventana(df, clientes, t0, limites_capping):
    """Ventana de observación limpia: saldo imputado, región completada y capping."""
    dv = ventana_observacion(df, clientes, t0)
    dv = imputar_saldo_promedio(dv)
    dv["Region"] = dv["client_id"].map(atributo_mas_cercano(df, "Region", t0)).fillna("Sin dato")
    if limites_capping:
        dv = aplicar_capping(dv, limites_capping)
    return dv
