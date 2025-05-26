import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generar_graficas_predictivas(df, modelo, feature_columns):
    """
    Función principal que genera gráficos predictivos utilizando el modelo entrenado.
    Parámetros:
        df (pd.DataFrame): DataFrame original sin codificar.
        modelo: modelo entrenado con método predict_proba.
        feature_columns (list): lista de columnas de entrada que usa el modelo (nombres tras get_dummies).
    """
    plot_pred_cancel_rate_by_antelacion(df, modelo, feature_columns)
    plot_pred_heatmap_tipo_plan(df, modelo, feature_columns)
    plot_pred_cancel_rate_by_canal(df, modelo, feature_columns)
    plot_pred_price_vs_invitados(df, modelo, feature_columns)


def plot_pred_cancel_rate_by_antelacion(df, modelo, feature_columns):
    """
    Gráfico de líneas predictivo: Probabilidad media de cancelación según días de antelación.
    """
    escenarios = sorted(df["dias_antelacion"].unique())
    resultados = []

    for d in escenarios:
        df_temp = df.copy()
        df_temp["dias_antelacion"] = d
        # Rellenar otras columnas con estadísticos
        for col in df.columns:
            if col in ["dias_antelacion", "cancelado"]:
                continue
            if df[col].dtype == "object":
                df_temp[col] = df[col].mode()[0]
            else:
                df_temp[col] = df[col].mean()
        # Codificar todo y alinear con feature_columns
        X = pd.get_dummies(df_temp)
        X = X.reindex(columns=feature_columns, fill_value=0)
        proba = modelo.predict_proba(X)[:, 1].mean()
        resultados.append({"dias_antelacion": d, "prob_cancel": proba})

    # Mostrar tabla

    df_res = pd.DataFrame(resultados)
    print(df_res.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x="dias_antelacion", y="prob_cancel", marker="o")
    plt.title("Probabilidad de Cancelación vs Días de Antelación")
    plt.xlabel("Días de Antelación")
    plt.ylabel("Probabilidad de Cancelación")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def plot_pred_heatmap_tipo_plan(df, modelo, feature_columns):
    """
    Heatmap predictivo de cancelaciones por Tipo de Cuarto vs Plan de Comidas.
    """
    tipos = df["tipo_cuarto"].unique()
    planes = df["plan_de_comidas"].unique()
    matrix = pd.DataFrame(index=tipos, columns=planes, dtype=float)

    for t in tipos:
        for p in planes:
            df_temp = df.copy()
            df_temp["tipo_cuarto"] = t
            df_temp["plan_de_comidas"] = p
            for col in df.columns:
                if col in ["tipo_cuarto", "plan_de_comidas", "cancelado"]:
                    continue
                if df[col].dtype == "object":
                    df_temp[col] = df[col].mode()[0]
                else:
                    df_temp[col] = df[col].mean()
            X = pd.get_dummies(df_temp)
            X = X.reindex(columns=feature_columns, fill_value=0)
            proba = modelo.predict_proba(X)[:, 1].mean()
            matrix.loc[t, p] = proba

    print(matrix.to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Probabilidad de Cancelación"},
    )
    plt.title("Predicción: Tipo de Cuarto vs Plan de Comidas")
    plt.xlabel("Plan de Comidas")
    plt.ylabel("Tipo de Cuarto")
    plt.show()


def plot_pred_cancel_rate_by_canal(df, modelo, feature_columns):
    """
    Gráfico de barras predictivo: Probabilidad de cancelación por Canal de Reserva.
    """
    canales = df["canal_de_reserva"].unique()
    datos = []

    for c in canales:
        df_temp = df.copy()
        df_temp["canal_de_reserva"] = c
        for col in df.columns:
            if col in ["canal_de_reserva", "cancelado"]:
                continue
            if df[col].dtype == "object":
                df_temp[col] = df[col].mode()[0]
            else:
                df_temp[col] = df[col].mean()
        X = pd.get_dummies(df_temp)
        X = X.reindex(columns=feature_columns, fill_value=0)
        proba = modelo.predict_proba(X)[:, 1].mean()
        datos.append({"canal": c, "prob_cancel": proba})

    df_bar = pd.DataFrame(datos)

    # Mostrar tabla
    print(df_bar.to_string(index=False))

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_bar, x="canal", y="prob_cancel")
    plt.title("Probabilidad de Cancelación por Canal de Reserva")
    plt.xlabel("Canal de Reserva")
    plt.ylabel("Probabilidad de Cancelación")
    plt.ylim(0, 1)
    plt.show()


def plot_pred_price_vs_invitados(df, modelo, feature_columns):
    """
    Scatter predictivo: Precio Total vs Invitados, coloreado por probabilidad de cancelación.
    """
    # Codificar todo el dataframe y alinear
    X_full = pd.get_dummies(df)
    X_full = X_full.reindex(columns=feature_columns, fill_value=0)
    proba = modelo.predict_proba(X_full)[:, 1]
    df_scatter = df[["invitados", "precio_total"]].copy()
    df_scatter["prob_cancel"] = proba

    # Mostrar tabla

    print(df_scatter.to_string(index=False))

    # Graficar
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["invitados"], df["precio_total"], c=proba, cmap="viridis", alpha=0.7
    )

    plt.colorbar(scatter, label="Probabilidad de Cancelación")
    plt.title("Precio Total vs Invitados (coloreado por prob. de cancelar)")
    plt.xlabel("Número de Invitados")
    plt.ylabel("Precio Total en dolares")
    plt.show()
