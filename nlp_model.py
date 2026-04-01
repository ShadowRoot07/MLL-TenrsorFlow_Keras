import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Datos de entrenamiento (Frases cortas)
sentences = [
    'me encanta este codigo',
    'esto es excelente y veloz',
    'que buen trabajo hiciste',
    'esto es horrible y lento',
    'no me gusta para nada',
    'error fatal en el sistema'
]
# Etiquetas: 1 (Positivo), 0 (Negativo)
labels = np.array([1, 1, 1, 0, 0, 0])

# 2. Tokenización
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convertir texto a secuencias de números
sequences = tokenizer.texts_to_sequences(sentences)
# 'Padding' asegura que todas las frases tengan la misma longitud (5 palabras)
padded = pad_sequences(sequences, maxlen=5, padding='post')

# 3. Modelo de NLP con Embedding
model = tf.keras.Sequential([
    # Capa de entrada: Convierte números en vectores de 16 dimensiones
    tf.keras.layers.Embedding(100, 16, input_length=5),
    # GlobalAveragePooling ayuda a resumir el sentimiento de la frase
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # 0 o 1
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("🚀 Entrenando Analizador de Sentimientos...")
model.fit(padded, labels, epochs=100, verbose=0)

# 4. Prueba con una frase que la IA NUNCA vio
test_phrase = ['este sistema es excelente']
test_seq = tokenizer.texts_to_sequences(test_phrase)
test_padded = pad_sequences(test_seq, maxlen=5, padding='post')

prediction = model.predict(test_padded)
print(f"\n📝 Frase: '{test_phrase[0]}'")
print(f"🔮 Sentimiento (0 Negativo / 1 Positivo): {prediction[0][0]:.4f}")

