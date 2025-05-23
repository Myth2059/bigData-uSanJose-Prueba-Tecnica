# clean.py
import pandas as pd


def cargar_y_limpiar_datos(ruta_archivo):
    df = pd.read_csv(ruta_archivo, sep=";")
    df.drop(columns=["nacionalidad_cliente"], inplace=True)

    # Convertir fechas
    df["fecha_reserva"] = pd.to_datetime(df["fecha_reserva"])
    df["fecha_check_in"] = pd.to_datetime(df["fecha_check_in"])

    # Convertir binarios
    df["huesped_repetido"] = df["huesped_repetido"].astype(int)
    df["cancelado"] = df["cancelado"].astype(int)

    df_sin_codificar = df.copy()

    # Codificación de variables categóricas
    df = pd.get_dummies(
        df,
        columns=[
            "tipo_cuarto",
            "plan_de_comidas",
            "canal_de_reserva",
        ],
        drop_first=True,
    )

    return df, df_sin_codificar
