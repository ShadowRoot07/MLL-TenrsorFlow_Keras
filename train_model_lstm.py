import tensorflow as tf
import numpy as np

# 1. Crear una secuencia (una simple onda de números)
# Queremos que la red aprenda que después de 1, 2, 3 viene el 4
def create_sequence(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Datos de ejemplo: Una serie del 1 al 100
data = np.linspace(0, 100, 100)
window_size = 3 # La red mirará 3 números para adivinar el 4to

X, y = create_sequence(data, window_size)

# IMPORTANTE: Ajustar dimensiones para LSTM -> (muestras, pasos_tiempo, características)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2. Modelo con Memoria (LSTM)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(window_size, 1)),
    # Capa LSTM con 50 unidades
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("🚀 Entrenando Red con Memoria (LSTM)...")
model.fit(X, y, epochs=200, verbose=0)

# 3. Prueba de fuego: ¿Qué viene después de [97, 98, 99]?
test_input = np.array([97, 98, 99]).reshape((1, window_size, 1))
prediction = model.predict(test_input)

print(f"\n🔮 Entrada: [97, 98, 99]")
print(f"🔮 Predicción del siguiente número: {prediction[0][0]:.2f}")
print(f"✅ El valor real debería ser 100.00")

