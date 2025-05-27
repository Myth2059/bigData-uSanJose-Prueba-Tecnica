# training.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os


def entrenar_modelo(df):
    # Separa características (X) y etiqueta (y)
    X = df.drop(
        ["id_reservacion", "fecha_reserva", "fecha_check_in", "cancelado"], axis=1
    )
    y = df["cancelado"]

    # Divide en datos de entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Crea y ajusta el modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Genera predicciones y mide precisión
    y_pred = modelo.predict(X_test)
    resultados = {
        "accuracy": accuracy_score(y_test, y_pred),
        "reporte": classification_report(y_test, y_pred, output_dict=True),
    }

    # Asegura que exista la carpeta para guardar el modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, "models/modelo_entrenado.pkl")  # Guarda el modelo

    return modelo, resultados
