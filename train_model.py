import tensorflow as tf
import numpy as np

# 1. Datos más complejos (X al cuadrado)
x_train = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
y_train = np.array([9, 4, 1, 0, 1, 4, 9], dtype=float) 

# 2. Modelo con CAPAS OCULTAS
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    # Capa con 8 neuronas y activación ReLU (para aprender la curva)
    tf.keras.layers.Dense(units=8, activation='relu'),
    # Otra capa para refinar el pensamiento
    tf.keras.layers.Dense(units=8, activation='relu'),
    # Salida única
    tf.keras.layers.Dense(units=1)
])

# 3. Compilar con un optimizador más inteligente (Adam)
model.compile(optimizer='adam', loss='mean_squared_error')

print("🚀 Entrenando red neuronal profunda...")
# Entrenamos por 1000 épocas porque el problema es más difícil
model.fit(x_train, y_train, epochs=1000, verbose=0)

# 4. Probar con un número que no estaba en el entrenamiento
print("🔮 Predicción para 5 (debería ser cercano a 25):")
print(model.predict(np.array([5.0])))

