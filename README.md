# 🧠 Análisis de Transacciones Financieras con Machine Learning

## 📋 Descripción del proyecto y objetivos

Este proyecto tiene como objetivo analizar un conjunto de datos de transacciones financieras para identificar patrones relevantes entre las variables, segmentar comportamientos y construir modelos de **Machine Learning** que permitan predecir el tipo de transacción o comportamiento financiero de los usuarios.  

Se busca comprender mejor la relación entre variables como el monto, el canal, la ocupación y el tipo de transacción, aportando valor en la **toma de decisiones para estrategias financieras o detección de comportamientos específicos**.

---

## 📊 Dataset utilizado y justificación de selección

El dataset contiene información sobre transacciones realizadas por distintos tipos de clientes, incluyendo variables como:

- Monto de transacción (`TransactionAmount`)  
- Tipo de transacción (`TransactionType`: Débito / Crédito)  
- Canal de uso (`Channel`: ATM, Online, Branch)  
- Ocupación del cliente (`Occupation`)  
- Saldo de la cuenta (`Balance`)  

Este conjunto de datos fue seleccionado por su **potencial para análisis predictivo** y por permitir el uso de múltiples técnicas de clasificación y visualización. Además, incluye tanto variables numéricas como categóricas, lo que lo hace ideal para aplicar algoritmos de Machine Learning supervisado.

---

## 💼 Problema de negocio abordado

El problema principal consiste en **predecir el tipo de transacción más probable (crédito o débito)** en función de las características del cliente y la operación, así como identificar patrones de comportamiento según el canal, monto y ocupación.  

Esta información puede ser útil para:
- Mejorar la personalización de servicios financieros.  
- Detectar anomalías o comportamientos atípicos.  
- Optimizar estrategias de marketing y canales de atención.  

---

## ⚙️ Metodología aplicada paso a paso

1. **Carga y limpieza de datos:**  
   - Detección y manejo de valores nulos.    

2. **Análisis exploratorio (EDA):**  
   - Visualización de distribuciones y relaciones entre variables.  
   - Análisis de correlaciones y detección de outliers.  

3. **División del dataset:**  
   - Separación en conjuntos de entrenamiento y prueba (80/20).  

4. **Entrenamiento de modelos de Machine Learning:**  
   - Entrenamiento con diferentes algoritmos y comparación de métricas.  

5. **Optimización de hiperparámetros:**  
   - Aplicación de *from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Definimos rangos en lugar de listas fijas
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': [5, 10, 15, None],
    'min_samples_split': randint(2, 10)
}

# Configurar búsqueda aleatoria
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,          # Número de combinaciones aleatorias a probar
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Entrenamiento
random_search.fit(X_train, y_train)* para encontrar la mejor combinación de parámetros.  

6. **Evaluación e interpretación de resultados:**  
   - Matrices de confusión, precisión, recall y F1-score.  
   - Análisis de importancia de características.  

---

## 🤖 Algoritmos de Machine Learning utilizados y justificación

Se entrenaron y compararon dos modelos principales:

- **Random Forest Classifier:**  
  Ideal para datos tabulares y con múltiples variables. Ofrece alta precisión y permite interpretar la importancia de características.  

- **Logistic Regression:**  
  Sencillo, rápido de entrenar y útil como modelo base para comparación.  

La elección de ambos algoritmos permite contrastar un modelo lineal y uno basado en árboles de decisión, logrando una evaluación más completa del rendimiento.

---

## 🔍 Principales hallazgos y patrones identificados

- Las **transacciones de tipo débito** son las más frecuentes, principalmente realizadas en **canales ATM**.  
- Las **transacciones de crédito** tienden a presentar **montos mayores**, especialmente en clientes con ocupación “estudiante”.  
- Las variables más influyentes en el modelo fueron:  
  `TransactionAmount`, `Occupation`, y `Channel`.  
- El modelo **Random Forest optimizado** presentó el mejor rendimiento general.

---

## 📎 Enlace al notebook en Google Colab

Acceso al notebook completo desde el siguiente enlace público:  
🔗 [Abrir en Google Colab](https://colab.research.google.com/drive/1YYdy_eSnl084aUn0D3A7nvPpWFFgtwS-?usp=sharing)

## 🧪 Instrucciones para reproducir el análisis

Clonar el repositorio o descargar los archivos del proyecto:
```
git clone https://github.com/tuusuario/proyecto-ml-transacciones.git
cd proyecto-ml-transacciones 
```

Abrir el notebook:

En Colab: sube el archivo .ipynb y el dataset.

En Jupyter:

jupyter notebook modelo_transacciones.ipynb

Ejecutar las celdas en orden para reproducir el análisis completo y las visualizaciones generadas.

## 📈 Conclusiones principales y recomendaciones

* El modelo Random Forest optimizado fue el más eficaz, alcanzando altos niveles de precisión en la clasificación de tipos de transacciones.

* Las variables más influyentes fueron TransactionAmount, Occupation y Channel, demostrando su relevancia para la predicción.

* Se recomienda implementar el modelo en un entorno de producción para detección temprana de fraudes o segmentación de clientes.

* Ampliar el dataset con variables temporales y demográficas puede mejorar la capacidad predictiva del modelo.

## ⚠️ Limitaciones identificadas

1. Desequilibrio de clases: el número de transacciones de tipo débito fue mucho mayor que las de crédito.

2. Falta de variables temporales: no se consideraron tendencias por fecha u hora.

3. Escalabilidad: el modelo aún no ha sido validado en datos de producción.

4. Simplificación del dataset: los datos son simulados, no necesariamente reflejan un entorno bancario real.