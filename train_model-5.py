import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Cargar y preparar (MNIST necesita una dimensión extra para el canal de color)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Redimensionar de (28,28) a (28,28,1) -> 1 canal de color (gris)
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0

# 2. Modelo CONVOLUCIONAL (El estándar de visión artificial)
model = models.Sequential([
    # Buscamos 32 patrones distintos de 3x3 píxeles
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Resumimos la imagen a la mitad
    layers.MaxPooling2D((2, 2)),
    # Buscamos 64 patrones más complejos
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Ahora sí, aplanamos para decidir
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # 'Dropout' ayuda a que no memorice, sino que aprenda
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 3. Compilar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Entrenando Red Convolucional (CNN)...")
model.fit(x_train, y_train, epochs=3)

print("\n🔍 Evaluación final de la CNN:")
model.evaluate(x_test, y_test, verbose=2)

