# main.py
from clean import cargar_y_limpiar_datos
from training import entrenar_modelo
from graphics import generar_graficas
from graficas_predictivas import lineplot_probabilidad


def main():
    # 1️⃣ Cargar y limpiar datos
    df, df_sin_codificar = cargar_y_limpiar_datos("data/reservas.csv")

    # 2️⃣ Entrenar el modelo
    modelo, resultados = entrenar_modelo(df)

    columnas_modelo = [
        "tipo_cuarto",
        "numero_noches",
        "numero_personas",
        "plan_comida",
        "canal_de_reserva",
        "precio_total",
        "dias_entre_reserva_y_checkin",
        "mes_reserva",
        "mes_checkin",
    ]

    lineplot_probabilidad(modelo, df, "tipo_cuarto", columnas_modelo)
    lineplot_probabilidad(modelo, df, "plan_comida", columnas_modelo)

    # 3️⃣ Graficar los resultados


# generar_graficas(df_sin_codificar, resultados)


if __name__ == "__main__":
    main()
