import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("graficos", exist_ok=True)


# 1️⃣ Gráfico de líneas: Probabilidad según una variable categórica u ordinal
def lineplot_probabilidad(modelo, df, variable, columnas_modelo):
    print(df.head())
    categorias = sorted(df[variable].unique())
    print(categorias)
    resultados = []

    for cat in categorias:
        df_temp = df.copy()
        df_temp[variable] = cat

        for col in columnas_modelo:
            if col != variable:
                if df[col].dtype == "object":
                    df_temp[col] = df[col].mode()[0]
                else:
                    df_temp[col] = df[col].mean()

        X = pd.get_dummies(df_temp[columnas_modelo])
        X = X.groupby(level=0, axis=1).sum()

        proba = modelo.predict_proba(X)[:, 1].mean()
        resultados.append({variable: cat, "probabilidad": proba})

    df_resultado = pd.DataFrame(resultados)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_resultado, x=variable, y="probabilidad", marker="o")
    plt.ylim(0, 1)
    plt.ylabel("Probabilidad de Cancelación")
    plt.title(f"Probabilidad de Cancelación según {variable}")
    plt.tight_layout()
    plt.savefig(f"graficos/lineplot_{variable}.png")
    plt.show()


# 2️⃣ Gráfico de cajas (boxplot): para variables numéricas


def boxplot_variable_numerica(df, variable_numerica):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="cancelado", y=variable_numerica)
    plt.title(f"{variable_numerica} vs Cancelación")
    plt.tight_layout()
    plt.savefig(f"graficos/boxplot_{variable_numerica}.png")
    plt.show()


# 3️⃣ Gráfico de violín (violinplot): distribución y densidad


def violinplot_variable(df, variable):
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="cancelado", y=variable)
    plt.title(f"Distribución de {variable} por Cancelación")
    plt.tight_layout()
    plt.savefig(f"graficos/violinplot_{variable}.png")
    plt.show()


# 4️⃣ Heatmap: para dos variables combinadas


def heatmap_cancelaciones(df, var1, var2):
    tabla = pd.crosstab(
        df[var1], df[var2], values=df["cancelado"], aggfunc="mean"
    ).fillna(0)

    plt.figure(figsize=(10, 7))
    sns.heatmap(tabla, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Probabilidad media de cancelación ({var1} vs {var2})")
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.tight_layout()
    plt.savefig(f"graficos/heatmap_{var1}_vs_{var2}.png")
    plt.show()
