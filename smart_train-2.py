import tensorflow as tf
import numpy as np

# Datos
x = np.random.rand(5000, 10) # Subimos a 5000 para que tenga más que aprender
y = (np.sum(x, axis=1) > 5).astype(int)
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(5000).batch(32)

# Modelo con DROPOUT
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    # Apagamos el 20% de las neuronas en cada paso
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

print("🚀 Entrenando Red Robusta con Dropout...")
model.fit(dataset, epochs=50, callbacks=[early_stop])

model.save('robust_shadow_model.keras')

