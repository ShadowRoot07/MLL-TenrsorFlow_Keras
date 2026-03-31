import tensorflow as tf
import numpy as np

# 1. Datos: Coordenadas (X, Y) y Etiqueta (0: Seguro, 1: Peligro)
# Digamos que si la suma de X+Y > 10, es PELIGRO.
x_data = np.array([
    [1, 2], [2, 3], [3, 1], [4, 4],   # Bajas (Seguro - 0)
    [7, 8], [9, 9], [10, 5], [8, 8]   # Altas (Peligro - 1)
], dtype=float)

y_data = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# 2. Modelo de Clasificación
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)), # Entran 2 números (X e Y)
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    # CAPA FINAL: 1 neurona con SIGMOIDE
    tf.keras.layers.Dense(1, activation='sigmoid') 
])

# 3. Compilar para Clasificación
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Entrenando Clasificador Shadow...")
model.fit(x_data, y_data, epochs=500, verbose=0)

# 4. Pruebas de fuego
test_seguro = np.array([[2, 2]])
test_peligro = np.array([[9, 10]])

print(f"🔮 Probabilidad de peligro para [2, 2]: {model.predict(test_seguro)[0][0]:.4f}")
print(f"🔮 Probabilidad de peligro para [9, 10]: {model.predict(test_peligro)[0][0]:.4f}")

