# Clasificación de Piezas Industriales mediante Deep Learning ⚙️📷

Este proyecto implementa un sistema de Visión Artificial integral diseñado para el control de calidad en el entorno de la Industria 4.0. Su objetivo es identificar y clasificar con alta precisión diversos componentes mecánicos pequeños (tornillos, tuercas, arandelas) del dataset *MVTec Screws* mientras se desplazan por una cinta transportadora industrial a una velocidad constante de 1 m/s.

El desafío se aborda desde una doble perspectiva: el correcto dimensionamiento del hardware óptico para congelar el movimiento, y el desarrollo de una arquitectura de software de baja latencia capaz de procesar imágenes en tiempo real.

## 🚀 Características Principales

### 1. Dimensionamiento Óptico (Hardware)

Para garantizar una resolución espacial de 135 píxeles para la pieza más pequeña (6 mm) a una distancia de trabajo de 500-700 mm:

* **Cámara:** Basler ace acA2440-75uc (5MP) con sensor **Global Shutter** para evitar deformaciones geométricas (*Rolling Shutter effect*).
* **Óptica:** Lente de 50 mm de focal fija (Montura C).
* **Iluminación y Exposición:** Uso de iluminación LED industrial de alta intensidad para permitir un tiempo de exposición ultracorto ($T_{exp} < 44.4 \mu s$), garantizando un *motion blur* inferior a 1 píxel a 1000 mm/s.

### 2. Preprocesamiento Avanzado y Curación de Datos

El éxito del modelo sobre un dataset reducido (385 imágenes) se apoya fuertemente en la ingeniería de datos:

* **Alineación Geométrica (Alignment):** Recorte y rotación mediante transformaciones afines para garantizar que todas las piezas se presenten verticalmente (Normalización Canónica).
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Mitigación de los brillos metálicos del acero para resaltar las texturas y geometrías críticas (como crestas de roscas).
* **Data Augmentation Estratégico:** Rotaciones, desplazamientos y volteos para dotar al modelo de robustez frente a errores de centrado en la cinta.

### 3. Arquitectura del Modelo de Deep Learning

Para cumplir con la exigencia de latencia mínima para operaciones en tiempo real, se descartaron redes pesadas en favor de una arquitectura ligera y altamente eficiente:

* **Backbone:** `MobileNetV2` (basada en *Inverted Residuals* y *Linear Bottlenecks*).
* **Cabezal de Clasificación (Top Layers):** Capa *Global Average Pooling 2D*, capa densa de 256 neuronas con activación *Swish*, y *Batch Normalization*.
* **Prevención de Overfitting:** Aplicación de Regularización L2 (0.001) y capas de *Dropout* (0.4 y 0.2).
* **Entrenamiento en 2 Fases:** Transfer Learning inicial congelando pesos de ImageNet, seguido de un Fine-Tuning selectivo con una tasa de aprendizaje de $10^{-5}$.

## 📊 Resultados y Rendimiento

El sistema ha sido validado superando los requisitos de un entorno de producción:

* **Accuracy Global:** 96% de precisión en la validación de 13 clases de tornillería.
* **F1-Score (Ponderado):** 0.95, demostrando un excelente equilibrio entre precisión y exhaustividad, incluso entre piezas con altísima similitud geométrica.
* **Viabilidad Técnica:** La latencia de inferencia de la arquitectura MobileNetV2 permite clasificar la pieza y actualizar el HUD industrial antes de que un nuevo objeto entre en el campo de visión de la cámara.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python
* **Deep Learning Framework:** TensorFlow / Keras (MobileNetV2)
* **Visión Artificial:** OpenCV (Procesamiento, CLAHE, Transformaciones Afines)
