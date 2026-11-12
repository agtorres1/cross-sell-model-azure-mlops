"""Entrena el modelo final y lo evalúa en test.

Uso:
    python -m src.train --datos data/raw/data.csv --t0 2019-01-01 --salida models/modelo.joblib
"""
import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data import cargar_base
from .metricas import resumen, tabla_deciles
from .modelo import ModeloPropension
from .universo import construir_universo


def separar_train_test(universo, test_size=0.20, random_state=42):
    """Split estratificado por cliente: 80 % train, 20 % test."""
    return train_test_split(universo, test_size=test_size,
                            stratify=universo["TGT"], random_state=random_state)


def main(datos, t0, salida):
    df = cargar_base(datos)
    universo = construir_universo(df, t0, con_target=True)
    train, test = separar_train_test(universo)

    modelo = ModeloPropension().entrenar(df, train, t0)

    lista_test = modelo.puntuar(df, test["client_id"], t0)
    y_test = test.set_index("client_id").loc[lista_test["client_id"], "TGT"].values
    metricas = {k: round(float(v), 4) for k, v in resumen(y_test, lista_test["puntaje"]).items()}

    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    modelo.guardar(salida)
    Path(salida).with_suffix(".metricas.json").write_text(json.dumps(metricas, indent=2))

    print(f"Universo: {len(universo)} clientes | train {len(train)} | test {len(test)}")
    print("Métricas en test:", metricas)
    print(tabla_deciles(y_test, lista_test["puntaje"]).round(3))
    return modelo, metricas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default="data/raw/data.csv")
    p.add_argument("--t0", default="2019-01-01")
    p.add_argument("--salida", default="models/modelo.joblib")
    a = p.parse_args()
    main(a.datos, a.t0, a.salida)
