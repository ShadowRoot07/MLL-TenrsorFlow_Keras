import tensorflow as tf
import numpy as np

# 1. Cargar el modelo pre-entrenado MobileNetV2 sin la capa final
# 'include_top=False' significa que quitamos la parte que clasifica
# y nos quedamos solo con la parte que "sabe ver"
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)

# 2. CONGELAR el modelo base
# No queremos cambiar lo que Google ya entrenó, solo queremos usarlo
base_model.trainable = False

# 3. Añadir nuestra propia "cabeza" al modelo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid') # Por ejemplo, para detectar solo una cosa
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print("\n🚀 ¡Modelo cargado! Este cerebro ya sabe reconocer patrones complejos.")

