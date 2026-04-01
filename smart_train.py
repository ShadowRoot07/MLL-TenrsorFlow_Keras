import tensorflow as tf
import numpy as np

# 1. Pipeline de datos (el mismo de la clase anterior)
x = np.random.rand(2000, 10)
y = (np.sum(x, axis=1) > 5).astype(int)
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(2000).batch(32)

# 2. Modelo
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. CONFIGURAR CALLBACKS (El Piloto Automático)
# Detener si el 'loss' no mejora durante 3 épocas seguidas
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

print("🚀 Entrenando con EarlyStopping (se detendrá solo si deja de mejorar)...")

# Pasamos el callback en una lista
model.fit(dataset, epochs=100, callbacks=[early_stop])

model.save('smart_shadow_model.keras')

