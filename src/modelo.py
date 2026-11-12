"""Modelo de propensión: preparación de variables + LightGBM en un solo objeto.

Se guarda todo junto para que producción use exactamente las mismas
transformaciones que entrenamiento (evita el training-serving skew).
"""
import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from .config import PARAMS_LGBM
from .features import PreparadorABT
from .metricas import asignar_decil


class ModeloPropension:

    def __init__(self, params=None):
        self.params = params or PARAMS_LGBM

    def entrenar(self, df, universo_train, t0):
        """Aprende las transformaciones y el modelo solo con los clientes de train."""
        self.preparador = PreparadorABT().fit(df, universo_train, t0)
        X = self.preparador.transform(df, universo_train["client_id"], t0)
        y = universo_train.set_index("client_id").loc[X.index, "TGT"]
        self.lgbm = LGBMClassifier(**self.params).fit(X, y)
        return self

    def puntuar(self, df, clientes, t0):
        """Lista de clientes con puntaje y decil, ordenada de mayor a menor."""
        X = self.preparador.transform(df, clientes, t0)
        lista = pd.DataFrame({
            "client_id": X.index,
            "puntaje": self.lgbm.predict_proba(X)[:, 1],
        })
        lista["decil"] = asignar_decil(lista["puntaje"])
        return lista.sort_values("puntaje", ascending=False).reset_index(drop=True)

    def guardar(self, ruta):
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta):
        return joblib.load(ruta)
