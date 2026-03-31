import tensorflow as tf
import numpy as np

# 1. Crear datos simples (X entrada, Y salida esperada)
# Por ejemplo: Y = 2X - 1
x_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_train = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 2. Definir el modelo (Keras Sequential)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. Compilar (Definir cómo aprende)
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. Entrenar
print("🚀 Entrenando neurona...")
model.fit(x_train, y_train, epochs=500, verbose=0)

# 5. Predecir algo nuevo
print("🔮 Predicción para el número 10 (debería ser casi 19):")
print(model.predict(np.array([10.0])))

