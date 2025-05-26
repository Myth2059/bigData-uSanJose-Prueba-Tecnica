# training.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os


def entrenar_modelo(df):
    X = df.drop(
        ["id_reservacion", "fecha_reserva", "fecha_check_in", "cancelado"], axis=1
    )
    y = df["cancelado"]
    # print(X.columns.tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    resultados = {
        "accuracy": accuracy_score(y_test, y_pred),
        "reporte": classification_report(y_test, y_pred, output_dict=True),
    }
    # Crear carpeta 'models' si no existe
    os.makedirs("models", exist_ok=True)

    # Guardar el modelo
    joblib.dump(modelo, "models/modelo_entrenado.pkl")

    return modelo, resultados
