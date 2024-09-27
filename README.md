# Music Genre Classification Project 🎶

## Introducción

Este proyecto se centra en la clasificación automática de géneros musicales utilizando redes neuronales. La música puede clasificarse en varios géneros (rock, pop, jazz, etc.), y esta tarea es fundamental en plataformas de streaming para mejorar las recomendaciones musicales. Utilizaremos técnicas de aprendizaje profundo y redes neuronales para entrenar un modelo que pueda predecir el género de una pieza musical a partir de sus características extraídas.

El modelo que implementamos buscará alcanzar altos niveles de precisión, experimentando con diferentes configuraciones y técnicas para mejorar el rendimiento en la clasificación de géneros.

## Problema

El problema central es crear un modelo capaz de clasificar correctamente las canciones en sus respectivos géneros musicales. El proceso se realiza analizando características extraídas de los archivos de audio, como espectrogramas y otros datos acústicos. Esta tarea presenta desafíos como la diversidad de géneros, los solapamientos entre ellos y las variaciones dentro de un mismo género.

La clasificación de géneros musicales es una tarea compleja debido a la naturaleza subjetiva de los géneros, donde algunas canciones pueden compartir características entre múltiples géneros. Además, la variabilidad en los ritmos, instrumentos y estilos introduce un nivel adicional de dificultad en la predicción.

## Dataset

Para este proyecto utilizamos el **GTZAN Music Genre Dataset** disponible en Kaggle, un dataset ampliamente utilizado en tareas de clasificación de géneros musicales. Este dataset contiene archivos de audio de 10 géneros distintos: **blues, classical, country, disco, hiphop, jazz, metal, pop, reggae y rock**.

El dataset incluye:

- **1000 pistas de audio** de 30 segundos cada una (100 por cada género).
- **Espectrogramas** de estas pistas, que proporcionan representaciones visuales de las frecuencias a lo largo del tiempo.
- **Características preprocesadas** que fueron extraídas de los archivos de audio.

### ¿Por qué 57 columnas?

El dataset contiene 57 columnas que corresponden a diversas características acústicas preprocesadas de los archivos de audio. Estas características fueron extraídas automáticamente utilizando herramientas de análisis de audio. Algunas de estas características incluyen:

- MFCC (Coeficientes Cepstrales en Frecuencia Mel) que son útiles para la identificación de las características tonales de la música.
- Chroma, Spectral Contrast, y otros parámetros que ayudan a capturar la estructura y el timbre de una pista musical.

Las 57 características son utilizadas como **entrada para el modelo de clasificación**.

### Exploración de varios modelos

Inicialmente, probaremos múltiples enfoques y arquitecturas de redes neuronales para encontrar la mejor combinación entre precisión, eficiencia y capacidad de generalización. Algunos de los experimentos que realizaremos incluyen:

- **Modelos simples vs. complejos**: Desde modelos con pocas capas y neuronas, hasta redes más profundas y complejas.
- **Redes neuronales completamente conectadas**: Utilizaremos una arquitectura básica de perceptrón multicapa (MLP) con funciones de activación ReLU y capas de normalización.
- **Ajuste de hiperparámetros**: Vamos a experimentar con varios parámetros como el tamaño de la red, el learning rate, y el número de épocas para entrenar el modelo.
- **Regularización**: Aplicaremos técnicas como dropout para reducir el sobreajuste, y batch normalization para estabilizar y acelerar el entrenamiento.


## Dependencias

A continuación, las principales librerías que debes instalar para poder ejecutar el proyecto:

- `torch`: Para la creación y entrenamiento del modelo de redes neuronales.
- `numpy`: Para operaciones numéricas y manejo de datos.
- `matplotlib`: Para generar visualizaciones de los resultados.
- `scikit-learn`: Para la evaluación del modelo.

Puedes instalar todas las dependencias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt


