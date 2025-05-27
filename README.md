# 📈 Predicción de Cancelación de Reservas Hoteleras

## 🎯 Introducción  
En este estudio queremos comprender **por qué** y **cuándo** los clientes cancelan sus reservaciones.  
Con esos conocimientos, el hotel podrá ajustar sus políticas y maximizar la ocupación.

## 💡 Justificación  
Cada cancelación deja una habitación vacía y pérdida de ingresos.  
Si podemos prever con antelación las cancelaciones, podremos:  
- 🔄 Ofrecer plazas liberadas a otros clientes.  
- 💰 Ajustar precios dinámicamente para llenar espacios.

---

## 📂 Estructura del Proyecto

| Archivo                 | Descripción                                                                                                         |
|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| `main.py`               | Orquestador principal: carga datos, entrena el modelo y genera gráficas predictivas.                                |
| `clean.py`              | Función `cargar_y_limpiar_datos()`: lectura, limpieza y codificación de variables.                                  |
| `training.py`           | Función `entrenar_modelo()`: divide datos, entrena el RandomForest y guarda el modelo con joblib.                   |
| `graphics.py`           | Funciones para generar las gráficas predictivas: líneas, heatmap, barras y scatter.                                |
| `data/reservas.csv`     | Dataset original de reservas.                                                                                       |
| `models/`               | Carpeta donde se almacena `modelo_entrenado.pkl` tras el entrenamiento.                                             |

---

## 🛠️ Tecnologías y Dependencias

| Herramienta / Librería | Versión mínima | Uso principal                                       |
|------------------------|---------------:|-----------------------------------------------------|
| Python                 | 3.7            | Lenguaje principal                                  |
| pandas                 | 1.2+           | Manejo y limpieza de datos                          |
| scikit-learn           | 0.24+          | Entrenamiento y evaluación del modelo               |
| matplotlib             | 3.3+           | Visualización de gráficos                           |
| seaborn                | 0.11+          | Mejoras estéticas en las visualizaciones            |
| joblib                 | 1.0+           | Serialización y carga del modelo entrenado          |

> **Tip:** También puedes usar un `requirements.txt`:
> ```txt
> pandas>=1.2
> scikit-learn>=0.24
> matplotlib>=3.3
> seaborn>=0.11
> joblib>=1.0
> ```

---

## 🚀 Instalación y Uso

1️⃣ Clona el repositorio  
```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```  

2️⃣ Instala dependencias  
```bash
pip install -r requirements.txt
```  

3️⃣ Ejecuta el script principal  
```bash
python main.py
```  
Verás en consola los resultados del entrenamiento y se generarán las gráficas en pantalla.

---

## 📊 Flujo de Trabajo

1️⃣ **Cargar y limpiar datos**  
```python
from clean import cargar_y_limpiar_datos

df, df_sin_codificar = cargar_y_limpiar_datos("data/reservas.csv")
```

2️⃣ **Entrenar el modelo**  
```python
from training import entrenar_modelo

modelo, resultados = entrenar_modelo(df)
print(resultados)
```

3️⃣ **Generar gráficas predictivas**  
```python
from graphics import generar_graficas_predictivas

feature_columns = [
    col for col in df.columns
    if col not in ["id_reservacion", "fecha_reserva", "fecha_check_in", "cancelado"]
]
generar_graficas_predictivas(df_sin_codificar, modelo, feature_columns)
```

Las gráficas generadas son:  
- 📈 Probabilidad de cancelación vs. días de antelación  
- 🔥 Heatmap: Tipo de cuarto vs. Plan de comidas  
- 📊 Barras: Probabilidad por canal de reserva  
- 🌐 Scatter: Precio total vs. invitados (coloreado por probabilidad)

---

## 🤝 Contribuciones  
¡Todas las contribuciones son bienvenidas!  
1. Haz un *fork* del proyecto (`git fork`)  
2. Crea tu branch (`git checkout -b feature/nueva-funcionalidad`)  
3. Haz tus cambios y *commit* (`git commit -m "Agrega nueva funcionalidad"`)  
4. Envía un *pull request*

---

## 📄 Licencia  
Este proyecto está bajo la licencia **MIT**.  
¡Disfruta y contribuye! 😊
