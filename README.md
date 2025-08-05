# TelecomX Churn Analysis

## Descripción
Este proyecto analiza la cancelación de clientes (churn) en una empresa de telecomunicaciones utilizando el dataset `TelecomX_Data_cleaned_transformed.csv`. Implementa un pipeline de machine learning en Python que utiliza Regresión Logística y Random Forest para predecir el churn, identifica factores clave (como `Antiguedad_Meses` y `Cargos_Mensuales`), y propone estrategias de retención. El pipeline incluye preprocesamiento (manejo de nulos, one-hot encoding, SMOTE), selección de características, modelado, evaluación de métricas (F1-Score, ROC-AUC), y visualizaciones detalladas (boxplots, scatter plots, matrices de correlación, curvas ROC).

**Características principales**:
- **Modelos**: Regresión Logística (F1: 0.69, ROC-AUC: 0.74) y Random Forest (F1: 0.74, ROC-AUC: 0.80).
- **Factores clave**: Menor antigüedad (~10 meses) y mayores cargos mensuales aumentan el churn.
- **Visualizaciones**: Boxplots, scatter plots, matrices de confusión, curvas ROC, y más.
- **Estrategias**: Descuentos para clientes nuevos, contratos a largo plazo, mejora en servicios de fibra óptica.

El proyecto está diseñado para ejecutarse en **Google Colab**, con explicaciones detalladas para principiantes y visualizaciones guardadas en la carpeta `figures`.

## Requisitos
- **Python 3.7+** (Google Colab incluye Python por defecto).
- **Librerías**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - IPython
- **Dataset**: `TelecomX_Data_cleaned_transformed.csv` contenido en el github
- **Google Colab**: Entorno recomendado para ejecución.

## Uso
1. **Abrir en Google Colab**:
   - Entra en google colab [Google Colab](https://colab.research.google.com/).
   - carga el archivo `telecomxparte2latam.ipynb` en una celda.

2. **Cargar el dataset**:
   - Sube `TelecomX_Data_cleaned_transformed.csv` a Colab (Archivos -> Subir).
   - Verifica con:
     ```python
     !ls
     ```

3. **Ejecutar el pipeline**:
   - Presiona **Shift + Enter** para ejecutar la celda.
   - El script:
     - Instala `imbalanced-learn`.
     - Procesa el dataset (elimina 11 nulos en `Cargos_Totales`, aplica one-hot encoding, SMOTE).
     - Entrena Regresión Logística y Random Forest.
     - Genera métricas (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
     - Crea visualizaciones en la carpeta `figures`.

4. **Descargar resultados**:
   - Visualizaciones: Ejecuta para comprimir la carpeta `figures`:
     ```python
     !zip -r figures.zip figures
     ```
     Descarga `figures.zip` desde la pestaña **Archivos**.
   - Archivos generados: Descarga `TelecomX_model_results.csv` (métricas) y `TelecomX_Data_modeled.csv` (dataset procesado).

## Estructura del Repositorio
```
TelecomX-Churn-Analysis/
│
├── churn_modeling_pipeline_logreg_rf_colab.py  # Script principal del pipeline
├── figures/                                   # Carpeta para visualizaciones (generada al ejecutar)
│   ├── class_distribution_before_smote.png
│   ├── class_distribution_after_smote.png
│   ├── boxplot_antiguedad_abandono.png
│   ├── boxplot_cargos_abandono.png
│   ├── scatter_antiguedad_cargos.png
│   ├── correlation_matrix_numeric.png
│   ├── correlation_matrix_all.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── coef_logreg.png
│   ├── feature_importance_rf.png
│   ├── roc_curves.png
├── TelecomX_model_results.csv                 # Métricas de los modelos (generado)
├── TelecomX_Data_modeled.csv                 # Dataset procesado (generado)
├── README.md                                 # Este archivo
```

## Resultados Clave
- **Desempeño de Modelos**:
  - **Regresión Logística**: F1-Score: 0.69, ROC-AUC: 0.74 (prueba). Simple e interpretable, pero con leve underfitting.
  - **Random Forest**: F1-Score: 0.74, ROC-AUC: 0.80 (prueba). Mejor desempeño, pero con overfitting (F1 entrenamiento: 0.95).
- **Factores Principales**:
  - `Antiguedad_Meses` (~ -0.35 correlación): Clientes nuevos (<12 meses) son más propensos a cancelar.
  - `Cargos_Mensuales` (~ 0.20 correlación): Altos cargos aumentan el churn.
  - `Tipo_Contrato_Mensual`: Contratos mensuales están asociados con mayor churn.
- **Visualizaciones**:
  - **Boxplots**: Confirman menor antigüedad (~10 meses) y cargos totales más bajos en clientes que cancelan.
  - **Matriz de Correlación**: Identifica `Antiguedad_Meses` y `Cargos_Mensuales` como variables clave.
  - **Curvas ROC**: Random Forest supera a Regresión Logística en capacidad predictiva.
- **Estrategias de Retención**:
  - Descuentos para clientes con <12 meses.
  - Ajustar cargos mensuales altos.
  - Promover contratos a largo plazo.
  - Mejorar calidad del servicio de fibra óptica.

## Explicación de Visualizaciones
- **Distribución de Abandono (Antes/Después de SMOTE)**: Muestra el desbalance inicial (~23% Abandono=1) y el balanceo tras SMOTE (50%/50%).
- **Boxplot de Antigüedad vs. Abandono**: Clientes que cancelan tienen menor antigüedad (~10 meses vs. ~30 meses para activos).
- **Boxplot de Cargos Totales vs. Abandono**: Clientes que cancelan tienen cargos totales más bajos, relacionados con menor antigüedad.
- **Scatter Plot de Antigüedad vs. Cargos Totales**: Clientes nuevos con cargos moderados-altos son más propensos a cancelar.
- **Matriz de Correlación (Numéricas)**: `Antiguedad_Meses` (-0.35) reduce churn, `Cargos_Mensuales` (0.20) lo aumenta.
- **Matriz de Confusión**: Muestra errores críticos (Falsos Negativos) para evaluar detección de churn.
- **Coeficientes de Regresión Logística**: Variables como `Antiguedad_Meses` (negativo) y `Cargos_Mensuales` (positivo) son clave.
- **Importancia de Variables (Random Forest)**: `Antiguedad_Meses` y `Cargos_Totales` son las más influyentes.
- **Curvas ROC**: Random Forest (AUC ~0.80) es mejor para distinguir clases.

## Licencia
[MIT License](LICENSE) 

## Contacto
Para preguntas o sugerencias, contacta a degartHub en GitHub 
