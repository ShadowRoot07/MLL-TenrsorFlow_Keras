import tensorflow as tf
from tensorflow.keras import layers, Model

# 1. Definimos la Entrada de forma explícita
inputs = tf.keras.Input(shape=(10,))

# 2. Conectamos las capas como funciones
# f(x) = y -> La capa recibe a la anterior entre paréntesis
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)

# 3. Definimos la Salida
outputs = layers.Dense(1, activation='sigmoid')(x)

# 4. Creamos el Modelo uniendo la Entrada y la Salida
model = Model(inputs=inputs, outputs=outputs, name="Shadow_Functional_Model")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary() # Verás una columna nueva llamada "Connected to"

class MyCustomLayer(layers.Layer):
    def __init__(self, units=32):
        super(MyCustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Aquí se crean los pesos (weights) de la capa
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer="random_normal",
                               trainable=True)

    def call(self, inputs):
        # Aquí va la lógica matemática (Ejemplo: producto punto)
        return tf.matmul(inputs, self.w)

