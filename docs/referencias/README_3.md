# Informe Técnico: Clasificación de Lenguaje de Señas mediante Perceptrón Multicapa (MLP)

> **Proyecto Reproducible de Machine Learning / Deep Learning**  
> **Dataset:** Sign Language MNIST (Kaggle)  
> **Autor/Equipo:** Técnicas Avanzadas de Machine Learning  
> **Estado:** Completado - Evaluación N°1

## Índice

1.  [Descripción del Problema de Negocio](#1-descripción-del-problema-de-negocio)
2.  [Objetivos del Proyecto](#objetivos-del-proyecto)
3.  [Definición de KPIs](#3-definición-de-kpis)
4.  [Descripción de las Fuentes de Datos Utilizadas](#4-descripción-de-las-fuentes-de-datos-utilizadas)
5.  [Preparación y Análisis Exploratorio de Datos (EDA)](#5-preparación-y-análisis-exploratorio-de-datos-eda)
6.  [Metodología Utilizada (CRISP-DM)](#6-metodología-utilizada-crisp-dm)
7.  [Arquitectura del Modelo y Entrenamiento](#arquitectura-del-modelo-y-entrenamiento)
8.  [Instrucciones de Reproducibilidad](#instrucciones-de-reproducibilidad)

## 1. Descripción del Problema de Negocio

### Contexto y Desafío

El **Lenguaje de Señas Americano (ASL - American Sign Language)** es uno de los medios primarios de comunicación gestual; sin embargo, la gran mayoría de las herramientas y plataformas automatizadas operan de manera exclusiva mediante texto o voz.

Para resolver este desfase de inclusión, surge la necesidad de contar con **sistemas informáticos capaces de reconocer e interpretar gestos manuales en tiempo real**, transformando las señales visuales en texto o comandos estructurados.

### Oportunidad de Negocio y Aplicación

El desarrollo de un módulo automatizado de clasificación gestual permite:

- **Mejorar la experiencia de cliente (CX):** Integración de kioscos autoservicio accesibles en sucursales bancarias, hospitales y estaciones de transporte.

- **Plataformas de Teleconferencia:** Generación automática de subtítulos o interpretación gestual en llamadas de video.

- **Aplicaciones Educativas:** Herramientas interactivas para el aprendizaje y validación de postura en el lenguaje de señas.

- **Sistemas Asistenciales e IoT:** Control de interfaces mediante gestos manuales simples.

### Alcance de la Solución (Variante del Problema)

Para responder a los requerimientos operacionales y tiempos de evaluación del proyecto, se definió un producto mínimo viable (MVP) enfocado en la clasificación estática de **6 clases gestuales clave**:

- **Clase 0 ('A')**, **Clase 1 ('C')**, **Clase 2 ('F')**, **Clase 3 ('K')**, **Clase 4 ('O')**, **Clase 5 ('Y')**.

Estas letras presentan configuraciones de la mano con variaciones distintivas y de alto valor representativo en la dactilología del lenguaje de señas.

## 2. Objetivos del Proyecto

### Objetivo General

Desarrollar y evaluar un modelo de Red Neuronal Artificial de tipo **Perceptrón Multicapa (MLP)** altamente reproducible para la clasificación automática de imágenes de gestos manuales del lenguaje de señas en 6 categorías objetivo, garantizando alta precisión y generalización.

### Objetivos Específicos

1.  **Garantizar Reproducibilidad Total:** Implementar un pipeline estructurado de código fijando semillas aleatorias (SEED = 42) tanto en NumPy como en TensorFlow/Keras.

2.  **Filtrado y Curaduría de Datos:** Extraer el subconjunto de datos correspondiente a las 6 clases de interés ('A', 'C', 'F', 'K', 'O', 'Y') desde el dataset global Sign Language MNIST y realizar la re-codificación de etiquetas a un rango contiguo ($0$ a $5$).

3.  **Análisis Exploratorio y Normalización (EDA):** Verificar la calidad e integridad de las imágenes (ausencia de valores nulos, distribución equilibrada de clases) y normalizar la intensidad de píxeles ($[0, 255] \rightarrow [0.0, 1.0]$).

4.  **Construcción y Entrenamiento del Modelo MLP:** Diseñar una arquitectura secuencial con capas densas ($784 \rightarrow 128 \text{ ReLU} \rightarrow 64 \text{ ReLU} \rightarrow 6 \text{ Softmax}$) optimizada con el algoritmo Adam.

5.  **Evaluación Holística de Métricas:** Medir el comportamiento de las curvas de costo (*Loss* vs *Val Loss*) y precisión (*Accuracy* vs *Val Accuracy*), además de matriz de confusión, *Precision*, *Recall* y *F1-Score*.

## 3. Definición de KPIs

Para determinar el éxito técnico y el impacto de negocio de la solución propuesta, se establecen los siguientes Indicadores Clave de Desempeño (KPIs):

### KPIs Técnicos / Rendimiento del Modelo

| **KPI Técnico**                     | **Métrica / Objetivo**         | **Descripción / Justificación**                                                                 |
|-------------------------------------|--------------------------------|-------------------------------------------------------------------------------------------------|
| **Global Test Accuracy**            | $\ge 90.0%$                  | Porcentaje global de predicciones correctas sobre el conjunto de prueba no visto.               |
| **Macro F1-Score**                  | $\ge 0.88$                   | Balance armónico entre *Precision* y *Recall* promediado entre las 6 clases para evitar sesgos. |
| **Generalización (Val Loss Gap)**   | $\vert\text{Train Loss} - \text{Val Loss}\vert$ | *(descripción no disponible en el documento original)* |
| **Tiempo de Inferencia (Latencia)** | $< 10\text{ ms per sample}$ | Tiempo máximo de cómputo por imagen para posibilitar integración en flujos en tiempo real.      |

### KPIs de Impacto de Negocio

| **KPI de Negocio**                   | **Objetivo Operacional**                  | **Impacto Esperado**                                                                         |
|--------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------|
| **Tasa de Error de Reconocimiento**  | $< 5.0%$ en clases objetivo            | Reducción drástica de interpretaciones erróneas en la interacción hombre-máquina.            |
| **Balance de Eficiencia / Recursos** | Despliegue en CPU sin requerir GPU pesada | Modelo liviano (~109,000 parámetros) de bajo costo computacional y fácil integración *Edge*. |
| **Inclusión y Accesibilidad UX**     | Cobertura exitosa de gestos frecuentes    | Habilitación de flujo de entrada de datos gestual rápido para usuarios con discapacidad.     |

## 4. Descripción de las Fuentes de Datos Utilizadas

### Origen de los Datos

El proyecto utiliza el dataset público **Sign Language MNIST**, alojado en Kaggle (datamunge/sign-language-mnist). Este conjunto representa un referente estándar derivado del formato del clásico MNIST, adaptado a la dactilología gestual.

### Estructura y Formato

- **Formato:** Archivos CSV tabulares (sign_mnist_train.csv y sign_mnist_test.csv).

- **Representación de Imagen:** Cada fila corresponde a una imagen en escala de grises de $28 \times 28$ píxeles, aplanada (*flattened*) en 784 columnas numéricas (pixel1 a pixel784).

- **Rango de Píxeles:** Entregados como enteros int64 con valores en el rango $[0, 255]$.

- **Variable Objetivo (label):** Enteros que representan letras del alfabeto en ASL (originalmente 24 clases, excluyendo 'J' y 'Z' por requerir movimiento dinámico).

### Dimensiones Iniciales vs Subconjunto Filtrado

- **Dataset Original:**

  - Entrenamiento: 27,455 registros y 785 columnas (1 label + 784 píxeles).

  - Prueba: 7,172 registros y 785 columnas.

- **Subconjunto Filtrado (6 Clases):**

  - Entrenamiento: **6,902 muestras**.

  - Prueba: **1,797 muestras**.

## 5. Preparación y Análisis Exploratorio de los Datos (EDA)

### 5.1 Control de Calidad de Datos

Se verificó la integridad del dataset previo a cualquier etapa de procesamiento:

- **Valores Faltantes (*Missing Values*):** $0$ nulos en el conjunto de entrenamiento y $0$ nulos en el conjunto de prueba.

- **Tipos de Datos:** Todos los atributos corresponden a enteros de 64 bits (int64), adecuados para procesamiento numérico matricial.

### 5.2 Mapeo y Recodificación de Etiqueta

Para trabajar con una salida categórica limpia y compatible con la función de pérdida sparse_categorical_crossentropy, se mapearon las letras originales a un entero contiguo entre 0 y 5:

| **Letra ASL** | **Clase Original (label)** | **Nueva Clase Codificada** | **Frecuencia en Train** | **Porcentaje (%)** |
|---------------|----------------------------|----------------------------|-------------------------|--------------------|
| **A**         | 0                          | **0**                      | 1,126                   | 16.31%             |
| **C**         | 2                          | **1**                      | 1,144                   | 16.57%             |
| **F**         | 5                          | **2**                      | 1,204                   | 17.44%             |
| **K**         | 10                         | **3**                      | 1,114                   | 16.14%             |
| **O**         | 14                         | **4**                      | 1,196                   | 17.33%             |
| **Y**         | 24                         | **5**                      | 1,118                   | 16.20%             |
| **Total**     | -                         | -                         | **6,902**               | **100.00%**        |

### 5.3 Análisis de Balance de Clases

Como se observa en la tabla, la clase con menor cantidad de imágenes es **'K' (1,114)** y la mayor es **'F' (1,204)**. El dataset presenta un **excelente balance de clases** (aproximadamente ~16.1% - 17.4% por clase), lo cual elimina la necesidad de aplicar técnicas agresivas de re-muestreo (SMOTE o *undersampling*).

### 5.4 Preprocesamiento y Normalización

1.  **Separación de Atributos y Etiqueta:** Separación de $X$ (píxeles 1 a 784) y $y$ (etiqueta codificada).

2.  **Escalamiento Min-Max:** Transformación de la escala de píxeles dividiendo por 255.0: $$X_{\text{norm}} = \frac{X}{255.0} \in [0.0, 1.0]$$ Esto acelera la convergencia del algoritmo de optimización (Adam) y previene la saturación de gradientes.

## 6. Metodología Utilizada (CRISP-DM)

El desarrollo del proyecto se estructuró rigurosamente bajo la metodología estándar de la industria **CRISP-DM (Cross-Industry Standard Process for Data Mining)**:

```
+-------------------------------------------------------------------------+
|                         CRISP-DM METODOLOGÍA                            |
|                                                                         |
|  [1. Comprensión del Negocio]  --->  [2. Comprensión de los Datos]     |
|               ^                                   |                    |
|               |                                   v                    |
|  [6. Despliegue] <--- [5. Evaluación] <--- [4. Modelado]               |
|                              ^                    |                     |
|                              +--- [3. Preparación |                    |
|                                     de Datos] <---+                    |
+-------------------------------------------------------------------------+
```

### Fases de Implementación en el Proyecto:

1.  **Comprensión del Negocio (*Business Understanding*):**

    - Identificación del problema de falta de accesibilidad gestual.

    - Definición de requerimientos técnicos y operacionales.

    - Selección de la variante de 6 clases para construcción del MVP.

2.  **Comprensión de los Datos (*Data Understanding*):**

    - Carga del conjunto de datos Sign Language MNIST.

    - Inspección de dimensiones ($28 \times 28$), rangos de valores y verificación de ausencia de nulos.

    - Análisis de la distribución de las letras seleccionadas.

3.  **Preparación de los Datos (*Data Preparation*):**

    - Filtrado del DataFrame para mantener solo las clases A, C, F, K, O, Y.

    - Re-etiquetado a números contiguos ($0, 1, 2, 3, 4, 5$).

    - Normalización de intensidad de píxeles ($/ 255.0$).

    - División de conjuntos en Train, Validation y Test.

4.  **Modelado (*Modeling*):**

    - Selección de la arquitectura perceptrón multicapa (MLP).

    - Capa de entrada de 784 neuronas.

    - Primera capa oculta: 128 neuronas con activación ReLU.

    - Segunda capa oculta: 64 neuronas con activación ReLU.

    - Capa de salida: 6 neuronas con activación Softmax para clasificación multiclase.

    - Configuración de la semilla aleatoria para asegurar reproducibilidad (SEED = 42).

5.  **Evaluación (*Evaluation*):**

    - Monitoreo de curvas de entrenamiento (*Loss* y *Accuracy*).

    - Evaluación de métricas en el conjunto de test aislado (*Precision*, *Recall*, *F1-score*).

    - Análisis del comportamiento de clasificación mediante la Matriz de Confusión.

    - Verificación del cumplimiento de los KPIs definidos.

6.  **Despliegue y Documentación (*Deployment*):**

    - Empaquetado del modelo y preparación del informe técnico en formato Markdown/README reproducible.

    - Definición de requerimientos de entorno e instrucciones para la ejecución.

## 7. Arquitectura del Modelo y Entrenamiento

### Resumen de la Arquitectura

- **Capa de Entrada:** Input(shape=(784,))

- **Capa Oculta 1:** Dense(128, activation='relu')

- **Capa Oculta 2:** Dense(64, activation='relu')

- **Capa de Salida:** Dense(6, activation='softmax')

```
[ Entrada: 784 píxeles ]
        │
        ▼
[ Dense: 128 neuronas + ReLU ]
        │
        ▼
[ Dense: 64 neuronas + ReLU ]
        │
        ▼
[ Salida: 6 neuronas + Softmax ] ──► Probabilidades (Clases 0-5: A, C, F, K, O, Y)
```

### Configuración del Entrenamiento

- **Función de Pérdida:** sparse_categorical_crossentropy

- **Optimizador:** Adam

- **Métrica Principal:** accuracy

- **Reproducibilidad:** SEED = 42 (configurada en Python, NumPy y TensorFlow).

## 8. Instrucciones de Reproducibilidad

Para ejecutar este proyecto de forma completamente reproducible, siga los siguientes pasos:

### 1. Requisitos de Entorno

- Python >= 3.9

- Librerías requeridas: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow

### 2. Ejecución Ficha de Código (Python/Notebook)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed

# Configuración estricta de reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
set_random_seed(SEED)
```
