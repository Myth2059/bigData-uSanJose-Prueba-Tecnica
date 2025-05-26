# main.py
from clean import cargar_y_limpiar_datos
from training import entrenar_modelo
from graphics import generar_graficas_predictivas
from joblib import load

# from graficas_predictivas import lineplot_probabilidad


def main():
    # 1️⃣ Cargar y limpiar datos
    df, df_sin_codificar = cargar_y_limpiar_datos("data/reservas.csv")

    # 2️⃣ Entrenar el modelo
    modelo, resultados = entrenar_modelo(df)
    print(resultados)
    # modelo = load("models/modelo_entrenado.pkl")

    feature_columns = [
        col
        for col in df.columns
        if col not in ["id_reservacion", "fecha_reserva", "fecha_check_in", "cancelado"]
    ]
    generar_graficas_predictivas(df_sin_codificar, modelo, feature_columns)


if __name__ == "__main__":
    main()
