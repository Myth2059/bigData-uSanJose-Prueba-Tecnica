# main.py
from clean import cargar_y_limpiar_datos
from training import entrenar_modelo
from graphics import generar_graficas_predictivas


def main():
    # 1️⃣ Leer y limpiar los datos
    df, df_sin_codificar = cargar_y_limpiar_datos("data/reservas.csv")

    # 2️⃣ Entrenar el modelo y obtener métricas
    modelo, resultados = entrenar_modelo(df)

    # 3️⃣ Definir columnas de entrada para las gráficas
    feature_columns = [
        col
        for col in df.columns
        if col not in ["id_reservacion", "fecha_reserva", "fecha_check_in", "cancelado"]
    ]

    # 4️⃣ Generar todas las visualizaciones predictivas
    generar_graficas_predictivas(df_sin_codificar, modelo, feature_columns)


if __name__ == "__main__":
    main()
