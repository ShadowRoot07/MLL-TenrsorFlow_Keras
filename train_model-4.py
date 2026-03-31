import tensorflow as tf
import numpy as np

# 1. Cargar el dataset MNIST (Números escritos a mano)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. NORMALIZACIÓN (Súper importante)
# Los píxeles van de 0 a 255. Los escalamos de 0 a 1 para que la red aprenda más rápido.
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Modelo de Visión
model = tf.keras.models.Sequential([
  # 'Flatten' convierte la matriz de 28x28 en una línea de 784 píxeles
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  # Capa de salida con 10 neuronas (una para cada número del 0 al 9)
  # Usamos 'softmax' para que el resultado sea una probabilidad entre todas las opciones
  tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Compilar para clasificación múltiple
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Entrenando Visión Artificial (MNIST)...")
# Entrenamos solo 5 épocas porque el dataset es enorme (60,000 imágenes)
model.fit(x_train, y_train, epochs=5)

# 5. Evaluar con datos que la red NUNCA ha visto
print("\n🔍 Evaluando precisión con datos de prueba:")
model.evaluate(x_test,  y_test, verbose=2)

