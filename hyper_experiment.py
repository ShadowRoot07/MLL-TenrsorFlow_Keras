import tensorflow as tf
import numpy as np

# Datos (usamos la misma lógica de 10k registros)
x = np.random.rand(10000, 10)
y = (np.sum(x, axis=1) > 5).astype(int)

# Función para crear diferentes modelos
def build_model(units, layers_count, dropout_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(10,)))
    
    for _ in range(layers_count):
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- EXPERIMENTO ---
# ¿Es mejor una red pequeña o una grande?
configs = [
    {"units": 16, "layers": 1, "name": "Pequeña"},
    {"units": 128, "layers": 3, "name": "Grande/Profunda"}
]

for config in configs:
    print(f"\n🧪 Probando configuración: {config['name']}")
    m = build_model(config['units'], config['layers'], 0.2)
    history = m.fit(x, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"✅ Resultado {config['name']}: Val_Accuracy = {final_val_acc:.4f}")

