import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Capas de Aumento de Datos (Esto se ejecuta dentro de la GPU/CPU)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

# 2. Modelo que usa MobileNetV2 + Aumento de Datos
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    data_augmentation, # <-- Aquí ocurre la magia
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Guardar
model.save('shadow_model_v2.keras')

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('shadow_model_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print("🚀 Modelo V2 con Data Augmentation guardado y listo.")

