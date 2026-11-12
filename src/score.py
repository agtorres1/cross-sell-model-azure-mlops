"""Puntúa el universo de un mes con un modelo ya entrenado.

Uso:
    python -m src.score --modelo models/modelo.joblib --datos data/raw/data.csv \
        --t0 2019-04-01 --salida data/scores/scores_2019-04.csv
"""
import argparse
from pathlib import Path

from .data import cargar_base
from .modelo import ModeloPropension
from .universo import construir_universo


def main(modelo, datos, t0, salida):
    m = ModeloPropension.cargar(modelo)
    df = cargar_base(datos)
    universo = construir_universo(df, t0, con_target=False)
    lista = m.puntuar(df, universo["client_id"], t0)
    lista.insert(0, "t0", t0)

    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    lista.to_csv(salida, index=False)
    print(f"{len(lista)} clientes puntuados para t0 = {t0} → {salida}")
    print(lista.groupby("decil")["puntaje"].agg(["count", "min", "max"]).round(3))
    return lista


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="models/modelo.joblib")
    p.add_argument("--datos", default="data/raw/data.csv")
    p.add_argument("--t0", required=True)
    p.add_argument("--salida", default=None)
    a = p.parse_args()
    main(a.modelo, a.datos, a.t0, a.salida or f"data/scores/scores_{a.t0[:7]}.csv")
