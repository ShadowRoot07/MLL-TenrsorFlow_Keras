import tensorflow as tf
import numpy as np

# 1. Generamos 100 puntos aleatorios entre -10 y 10
x_train = np.linspace(-10, 10, 100)
# La regla es X al cuadrado
y_train = x_train**2

# 2. Modelo con más capacidad
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    # Aumentamos a 16 neuronas
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Usamos una tasa de aprendizaje (learning rate) específica para Adam
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')

print("🚀 Entrenando con 100 puntos de datos...")
# Bajamos las épocas a 500 porque ahora tiene más datos para aprender rápido
model.fit(x_train, y_train, epochs=500, verbose=0)

# 3. Probamos con el 5 (que está DENTRO del rango -10 a 10)
print("🔮 Predicción para 5 (debería ser casi 25):")
print(model.predict(np.array([5.0])))

# 4. Probamos con el 9 (que también está dentro)
print("🔮 Predicción para 9 (debería ser casi 81):")
print(model.predict(np.array([9.0])))

