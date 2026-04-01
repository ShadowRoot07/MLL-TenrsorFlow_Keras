import tensorflow as tf
import numpy as np

# 1. Generamos 10,000 datos para tener una buena muestra
x_all = np.random.rand(10000, 10)
y_all = (np.sum(x_all, axis=1) > 5).astype(int)

# 2. SPLIT: 80% para entrenar, 20% para el examen final (test)
split = int(0.8 * len(x_all))
x_train, x_test = x_all[:split], x_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# 3. Pipelines separados
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(8000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 4. Modelo Robusto
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Entrenamiento con VALIDACIÓN
print("🚀 Entrenando y evaluando en tiempo real...")
model.fit(train_ds, epochs=20, validation_data=test_ds) # <-- Mira esto

# 6. Evaluación Final (El Examen)
print("\n📝 REALIZANDO EXAMEN FINAL...")
results = model.evaluate(test_ds)
print(f"Resultado del examen - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")

