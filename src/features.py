"""Construcción de la ABT (03_abt) y transformaciones que aprenden de train.

Dos partes:
  - construir_abt(): todo lo determinístico (agregados, flags, fechas).
  - PreparadorABT: lo que se aprende con train y se guarda para producción
    (límites del capping, modas, tasas del target encoding).
"""
import numpy as np
import pandas as pd

from .cleaning import limpiar_ventana, tipo_de_tarjeta, imputar_saldo_promedio, ventana_observacion
from .config import (AGG_CAJA, AGG_CANALES, AGG_TARJETA, CAJAS, FEATURES,
                     MESES_RECIENTES, MIN_CLIENTES_CATEGORIA, ORDEN_EDAD,
                     PRODUCTOS, SEGUROS, VARS_CAPPING, YES_NO)
from .data import mes, meses


def agregar(dv, columnas, sufijo=""):
    """Resume los meses de cada variable por cliente: suma, máx, mín, promedio,
    mediana, valores distintos y meses con valor distinto de 0."""
    agg = dv.groupby("client_id")[columnas].agg(["sum", "max", "min", "mean", "median", "nunique"])
    agg.columns = ["_".join(c) + sufijo for c in agg.columns]
    no_cero = (dv[columnas] != 0).groupby(dv["client_id"]).sum()
    no_cero.columns = [c + "_meses_no_cero" + sufijo for c in no_cero.columns]
    return agg.join(no_cero)


def _a_binaria(serie):
    return serie.map({"Yes": 1, "No": 0})


def _variables_foto(dv, t0):
    """Variables de la foto en t0: productos, fechas, edad, sexo y ratios."""
    tf = dv[dv["Month"] == mes(t0)].set_index("client_id").copy()
    out = pd.DataFrame(index=tf.index)

    out["SavingAccount_Balance_Average"] = tf["SavingAccount_Balance_Average"]
    for col in YES_NO:
        out[col] = _a_binaria(tf[col])

    # Seguros: se toman del último mes de la ventana que tenga datos cargados
    hay_seguros = dv.groupby("Month")[SEGUROS].apply(lambda x: (x == "Yes").any().any())
    mes_seguros = hay_seguros[hay_seguros].index.max() if hay_seguros.any() else mes(t0)
    seg = dv[dv["Month"] == mes_seguros].set_index("client_id")[SEGUROS]
    for col in SEGUROS:
        out[col] = _a_binaria(seg[col]).reindex(out.index)

    out["SavingAccount_Credits_Amounts"] = tf["SavingAccount_Credits_Amounts"]
    out["SavingAccount_Debits_Amounts"] = tf["SavingAccount_Debits_Amounts"]

    # Historia con el banco
    corte = pd.Timestamp(mes(t0))
    primero = pd.to_datetime(tf["First_product_dt"])
    ultimo = pd.to_datetime(tf["Last_product_dt"])
    out["Meses_Antiguedad"] = ((corte - primero).dt.days / 30).clip(lower=0)
    out["Meses_Ultimo_Producto"] = ((corte - ultimo).dt.days / 30).clip(lower=0)
    out["Meses_Primer_Ultimo"] = (ultimo - primero).dt.days / 30
    out["Flag_Contrato_Reciente"] = (out["Meses_Ultimo_Producto"] <= 6).astype(int)
    out["Flag_Producto_Unico"] = (primero == ultimo).astype(int)
    out["Flag_Cliente_Nuevo"] = (out["Meses_Antiguedad"] <= 12).astype(int)

    # Qué tiene
    out["Cant_Seguros"] = out[SEGUROS].sum(axis=1)
    out["Flag_Tiene_Seguro"] = (out["Cant_Seguros"] > 0).astype(int)
    out["Cant_Productos"] = out[PRODUCTOS].sum(axis=1) + out["Flag_Tiene_Seguro"]
    out["Cant_Cajas"] = out[CAJAS].sum(axis=1)
    out["Flag_Falta_Tarjeta"] = ((out["Cant_Cajas"] > 0) & (out["CreditCard_Active"] == 0)).astype(int)
    out["Flag_Falta_Prestamo"] = ((out["CreditCard_Active"] == 1) & (out["Loan_Active"] == 0)).astype(int)
    out["Productos_por_Anio"] = out["Cant_Productos"] / np.maximum(out["Meses_Antiguedad"] / 12, 1)

    # Cómo maneja la plata
    creditos, debitos = out["SavingAccount_Credits_Amounts"], out["SavingAccount_Debits_Amounts"]
    total = creditos + debitos
    out["Prop_Creditos"] = np.where(total == 0, 0.5, creditos / total.where(total != 0, 1))
    out["Saldo_Neto"] = creditos - debitos
    out["Flag_Movimientos"] = ((creditos > 0) | (debitos > 0)).astype(int)
    out["Contactabilidad"] = out["Mobile"] + out["Email"]

    # Demográficas
    out["Age_grp_ord"] = tf["Client_Age_grp"].map(ORDEN_EDAD)
    out["Sex_F"] = (tf["Sex"] == "F").astype(int)
    out["Region"] = tf["Region"]          # texto: se codifica en PreparadorABT
    return out


def _evolucion_revolving(dv, t0):
    """Cambio del saldo financiado de la tarjeta entre t0, t0-2 y t0-5."""
    rev = dv.pivot(index="client_id", columns="Month", values="CreditCard_Revolving")
    m0, m2, m5 = rev[mes(t0)], rev[mes(t0, -2)], rev[mes(t0, -5)]
    comp = pd.DataFrame(index=rev.index)
    comp["Rev_0vs2_abs"] = m0 - m2
    comp["Rev_0vs5_abs"] = m0 - m5
    comp["Rev_2vs5_abs"] = m2 - m5
    comp["Rev_0vs2_dir"] = np.sign(m0 - m2)
    comp["Rev_0vs5_dir"] = np.sign(m0 - m5)
    comp["Rev_2vs5_dir"] = np.sign(m2 - m5)
    return comp


def _meses_con_tarjeta(dv):
    cc = dv[["client_id"]].copy()
    cc["Meses_con_tarjeta"] = (dv["CreditCard_Active"] == "Yes").astype(int)
    cc["Meses_premium"] = (dv["CreditCard_Premium"] == "Yes").astype(int)
    return cc.groupby("client_id").sum()


def construir_abt(df, clientes, t0, limites_capping=None):
    """ABT de una fila por cliente, sin codificar las categóricas.

    Devuelve todas las variables de 03_abt más dos columnas de texto,
    'Region' y 'CreditCard_Product', que codifica PreparadorABT.
    """
    dv = limpiar_ventana(df, clientes, t0, limites_capping)
    recientes = dv[dv["Month"].isin(meses(t0, -(MESES_RECIENTES - 1), 0))]

    foto = dv[dv["Month"] == mes(t0)]
    tarjeta = tipo_de_tarjeta(df, foto, t0).rename("CreditCard_Product")

    partes = [
        _variables_foto(dv, t0),
        tarjeta,
        agregar(dv, AGG_CAJA),
        agregar(dv, AGG_TARJETA),
        agregar(dv, AGG_CANALES),
        agregar(recientes, AGG_TARJETA, sufijo="_u3m"),
        _evolucion_revolving(dv, t0),
        _meses_con_tarjeta(dv),
    ]
    abt = pd.concat(partes, axis=1).copy()
    abt["CreditCard_Balance_ARG_max_0"] = (abt["CreditCard_Balance_ARG_max"] > 0).astype(int)
    abt.index.name = "client_id"
    return abt.loc[sorted(clientes)]


class PreparadorABT:
    """Arma la matriz de variables del modelo a partir de la base mensual.

    fit() aprende con los clientes de train:
      - límites del capping (p95 de valores positivos),
      - región más frecuente (para 'Sin dato'),
      - tipo de tarjeta más frecuente (para tarjetas sin tipo informado),
      - tipos de tarjeta con pocos clientes (se agrupan en 'Otros'),
      - tasa de target por región y por tipo de tarjeta.
    transform() aplica exactamente esos números a cualquier mes y clientes.
    """

    def fit(self, df, universo_train, t0):
        clientes = universo_train["client_id"]
        tgt = universo_train.set_index("client_id")["TGT"]

        dv = imputar_saldo_promedio(ventana_observacion(df, clientes, t0))
        self.limites_capping_ = {
            col: float(dv.loc[dv[col] > 0, col].quantile(0.95)) for col in VARS_CAPPING
        }

        abt = construir_abt(df, clientes, t0, self.limites_capping_)

        region = abt["Region"]
        self.moda_region_ = region[region != "Sin dato"].mode()[0]
        region = region.replace("Sin dato", self.moda_region_)
        self.tasa_region_ = tgt.groupby(region).mean().to_dict()

        tarjeta = abt["CreditCard_Product"]
        self.moda_tarjeta_ = tarjeta[tarjeta != "No tiene"].mode()[0]
        tarjeta = tarjeta.fillna(self.moda_tarjeta_)
        conteo = tarjeta.value_counts()
        self.tarjetas_conocidas_ = set(conteo[conteo >= MIN_CLIENTES_CATEGORIA].index)
        tarjeta = self._agrupar_tarjeta(tarjeta)
        self.tasa_tarjeta_ = tgt.groupby(tarjeta).mean().to_dict()

        self.tasa_global_ = float(tgt.mean())
        self.t0_entrenamiento_ = t0
        return self

    def _agrupar_tarjeta(self, tarjeta):
        return tarjeta.where(tarjeta.isin(self.tarjetas_conocidas_), "Otros")

    def transform(self, df, clientes, t0, completa=False):
        """Matriz de variables para `clientes` en la foto t0.

        completa=False devuelve solo las 50 variables del modelo (FEATURES);
        completa=True devuelve la ABT entera ya codificada.
        """
        abt = construir_abt(df, clientes, t0, self.limites_capping_).copy()

        region = abt.pop("Region").replace("Sin dato", self.moda_region_)
        abt["Region_pct_target"] = region.map(self.tasa_region_).fillna(self.tasa_global_)

        tarjeta = self._agrupar_tarjeta(abt.pop("CreditCard_Product").fillna(self.moda_tarjeta_))
        abt["Tarj_pct_target"] = tarjeta.map(self.tasa_tarjeta_).fillna(self.tasa_global_)

        return abt if completa else abt[FEATURES]
