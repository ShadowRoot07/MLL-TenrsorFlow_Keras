import tensorflow as tf
import os

# (Usamos el mismo modelo de la clase anterior)
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False)
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --- NUEVO: GUARDADO DEL MODELO ---

# 1. Guardar en formato Keras estándar
model.save('shadow_model.keras')
print("💾 Modelo guardado como shadow_model.keras")

# 2. Convertir a TensorFlow Lite (Para tu celular)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Escribir el archivo .tflite
with open('shadow_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("📱 Modelo optimizado para móvil guardado como shadow_model.tflite")

