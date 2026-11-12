"""Constantes del proyecto: definición de ventanas, columnas y variables finales.

Todo lo que es una *decisión* (qué columnas se agregan, qué variables usa el
modelo, cuántos meses mira) vive acá, en un solo lugar. Las funciones de
src/ lo importan; los notebooks y Azure también.
"""

# --- Ventanas temporales (en meses, relativas a t0) -------------------------
# t0 es el último mes de la ventana de observación ("la foto").
MESES_OBSERVACION = 6          # t0-5 ... t0
MESES_RECIENTES = 3            # t0-2 ... t0 (agregados "_u3m")
MESES_LEAD = 1                 # t0+1: tiempo para armar la lista y contactar
MESES_RESPUESTA = 2            # t0+2 ... t0+3: ventana donde se mide el target

# --- Columnas de la base -----------------------------------------------------
YES_NO = [
    "CreditCard_Premium", "CreditCard_Active", "Loan_Active", "Mortgage_Active",
    "SavingAccount_Active_ARG_Salary", "SavingAccount_Active_ARG",
    "SavingAccount_Active_DOLLAR", "DebitCard_Active", "Investment_Active",
    "Mobile", "Email",
]

SEGUROS = [
    "Insurance_Life", "Insurance_Home", "Insurance_Accidents",
    "Insurance_Mobile", "Insurance_ATM", "Insurance_Unemployment",
]

PRODUCTOS = [
    "Loan_Active", "CreditCard_Active", "Mortgage_Active",
    "SavingAccount_Active_ARG", "SavingAccount_Active_DOLLAR",
    "DebitCard_Active", "Investment_Active",
]

CAJAS = ["SavingAccount_Active_ARG", "SavingAccount_Active_DOLLAR",
         "SavingAccount_Active_ARG_Salary"]

ORDEN_EDAD = {
    "Entre 18 y 29 años": 1,
    "Entre 30 y 39 años": 2,
    "Entre 40 y 49 años": 3,
    "Entre 50 y 59 años": 4,
    "Entre 60 y 64 años": 5,
    "Entre 65 y 69 años": 6,
    "Mayor a 70 años": 7,
}

# Variables con capping en el p95 de los valores positivos (02_limpieza)
VARS_CAPPING = [
    "SavingAccount_Balance_Average",
    "SavingAccount_Transfer_Out_Amount",
    "SavingAccount_Transfer_In_Amount",
]

# Tipos de tarjeta con menos clientes que esto se agrupan en "Otros"
MIN_CLIENTES_CATEGORIA = 100

# Variables agregadas sobre la ventana de observación (03_abt)
AGG_CAJA = [
    "SavingAccount_Balance_LastDate", "SavingAccount_Balance_Average",
    "SavingAccount_Days_with_use", "SavingAccount_Transactions_Transactions",
    "SavingAccount_Salary_Payment_Amount", "SavingAccount_Transfer_In_Amount",
    "SavingAccount_Transfer_Out_Amount", "SavingAccount_ATM_Extraction_Amount",
    "SavingAccount_Service_Payment_Amount", "SavingAccount_CreditCard_Payment_Amount",
    "SavingAccount_DebitCard_Spend_Amount", "SavingAccount_Credits_Amounts",
    "SavingAccount_Debits_Amounts",
]
AGG_TARJETA = [
    "CreditCard_Balance_ARG", "CreditCard_Total_Limit", "CreditCard_Total_Spending",
    "CreditCard_Spending_1_Installment", "CreditCard_Spending_Installments",
    "CreditCard_Revolving", "CreditCard_Payment_Aut_Debit",
]
AGG_CANALES = [
    "Operations_Bank", "Operations_HomeBanking", "Operations_Mobile",
    "Operations_ATM", "Investment_Numbers",
]

# --- Variables finales del modelo (salida de 04_seleccion) -------------------
FEATURES = [
    "Productos_por_Anio", "Meses_Antiguedad", "Meses_Ultimo_Producto", "Age_grp_ord",
    "Rev_0vs2_abs", "Tarj_pct_target", "CreditCard_Balance_ARG_max_0",
    "CreditCard_Total_Limit_min", "Rev_0vs5_abs", "SavingAccount_Balance_LastDate_min",
    "Rev_2vs5_abs", "Meses_Primer_Ultimo", "CreditCard_Total_Spending_min_u3m",
    "Saldo_Neto", "CreditCard_Revolving_min", "SavingAccount_Debits_Amounts",
    "CreditCard_Spending_Installments_max", "CreditCard_Total_Spending_min",
    "CreditCard_Spending_1_Installment_max", "SavingAccount_Balance_LastDate_median",
    "CreditCard_Balance_ARG_min", "Region_pct_target", "CreditCard_Total_Spending_max_u3m",
    "CreditCard_Revolving_median", "CreditCard_Revolving_max_u3m", "Prop_Creditos",
    "CreditCard_Total_Spending_mean", "SavingAccount_Balance_Average",
    "SavingAccount_Credits_Amounts_max", "SavingAccount_Credits_Amounts_min",
    "CreditCard_Total_Spending_max", "CreditCard_Spending_Installments_min",
    "SavingAccount_Transfer_In_Amount_mean", "CreditCard_Revolving_median_u3m",
    "SavingAccount_Balance_LastDate_max", "CreditCard_Spending_Installments_min_u3m",
    "SavingAccount_Days_with_use_mean", "SavingAccount_Balance_Average_max",
    "Operations_HomeBanking_max", "SavingAccount_Balance_Average_min",
    "SavingAccount_Transactions_Transactions_mean", "CreditCard_Spending_Installments_mean",
    "CreditCard_Revolving_max", "SavingAccount_Credits_Amounts_mean",
    "CreditCard_Spending_1_Installment_mean", "SavingAccount_CreditCard_Payment_Amount_median",
    "CreditCard_Revolving_min_u3m", "CreditCard_Spending_1_Installment_max_u3m",
    "SavingAccount_Balance_Average_sum", "SavingAccount_CreditCard_Payment_Amount_max",
]

# --- Modelo final (05_modelado, paso 5) --------------------------------------
PARAMS_LGBM = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 16,
    "min_child_samples": 148,
    "subsample": 0.81,
    "subsample_freq": 1,
    "colsample_bytree": 0.76,
    "random_state": 42,
    "verbose": -1,
}
