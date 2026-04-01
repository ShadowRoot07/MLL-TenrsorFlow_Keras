import tensorflow as tf
import numpy as np

# 1. Simulemos una base de datos de 1000 registros (X, Y)
# Imagina que estos vienen de tu PostgreSQL
def load_mock_data():
    x = np.random.rand(1000, 10) # 1000 ejemplos de 10 características
    y = (np.sum(x, axis=1) > 5).astype(int) # Regla: si suma > 5, es 1
    return x, y

x_raw, y_raw = load_mock_data()

# 2. Creamos el Pipeline de TensorFlow
# Esto es mucho más rápido que pasarle los arrays directamente
dataset = tf.data.Dataset.from_tensor_slices((x_raw, y_raw))

# Configuramos la "Cinta Transportadora":
dataset = dataset.shuffle(buffer_size=1000) # Mezcla los datos
dataset = dataset.batch(32)                # Entrega grupos de 32
dataset = dataset.prefetch(tf.data.AUTOTUNE) # Prepara el siguiente grupo mientras la CPU entrena

# 3. Modelo Simple para probar el Pipeline
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Entrenando con un Pipeline de Datos eficiente...")
# Ahora pasamos el objeto 'dataset' directamente
model.fit(dataset, epochs=10)

# Guardamos el resultado
model.save('pipeline_model.keras')
print("✅ Modelo entrenado con Pipeline guardado.")

