# graphics.py
import matplotlib.pyplot as plt
import seaborn as sns

def generar_graficas(df, resultados):
    sns.set(style="whitegrid")

    # 1️⃣ Cancelaciones por canal de reserva
    plt.figure(figsize=(10, 5))
    sns.countplot(x="canal_de_reserva", hue="cancelado", data=df)
    plt.title("Cancelaciones por canal de reserva")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cancelaciones_por_canal.png")

    # 2️⃣ Cancelaciones por tipo de habitación
    plt.figure(figsize=(10, 5))
    sns.countplot(x="tipo_cuarto", hue="cancelado", data=df)
    plt.title("Cancelaciones por tipo de habitación")
    plt.tight_layout()
    plt.savefig("cancelaciones_por_tipo_cuarto.png")

    # 3️⃣ Precio total vs cancelación
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="cancelado", y="precio_total", data=df)
    plt.title("Precio total vs Cancelación")
    plt.tight_layout()
    plt.savefig("precio_vs_cancelacion.png")

    # 4️⃣ Resultados del modelo
    print("📊 Precisión del modelo:", resultados["accuracy"])
    print("🧾 Reporte de clasificación:")
    for clase, metrica in resultados["reporte"].items():
        if isinstance(metrica, dict):
            print(f"{clase}: Precision={metrica['precision']:.2f}, Recall={metrica['recall']:.2f}, F1={metrica['f1-score']:.2f}")
