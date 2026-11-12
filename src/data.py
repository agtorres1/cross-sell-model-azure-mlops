"""Carga de la base mensual y manejo de meses."""
import pandas as pd


def cargar_base(ruta):
    """Lee el CSV crudo (separado por '|') y corrige client_id.

    Hay filas con client_id no numérico al final del archivo: se descartan.
    """
    df = pd.read_csv(ruta, sep="|", low_memory=False)
    df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce")
    df = df[df["client_id"].notna()].copy()
    df["client_id"] = df["client_id"].astype(int)
    df["Month"] = df["Month"].astype(str)
    return df


def mes(t0, desplazamiento=0):
    """Devuelve el mes t0 + desplazamiento como texto 'AAAA-MM-01'.

    >>> mes("2019-01-01", -5)
    '2018-08-01'
    """
    periodo = pd.Period(t0, freq="M") + desplazamiento
    return periodo.start_time.strftime("%Y-%m-%d")


def meses(t0, desde, hasta):
    """Lista de meses entre t0+desde y t0+hasta, inclusive."""
    return [mes(t0, k) for k in range(desde, hasta + 1)]
