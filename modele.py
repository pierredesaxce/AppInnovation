import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Set the GPU device as the default device
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU.")

# Now you can build and train your model as usual
# ...

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    passwords = [line.strip() for line in file]

# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Préparation des données d'entraînement
max_len = max([len(password) for password in passwords])
X = np.zeros((len(passwords), max_len, len(chars)), dtype=np.bool_)
y = np.zeros((len(passwords), max_len, len(chars)), dtype=np.bool_)

for i, password in enumerate(passwords):
    for t, char in enumerate(password):
        X[i, t, char_indices[char]] = 1
        if t < max_len - 1:
            y[i, t + 1, char_indices[char]] = 1

# Création du modèle RNN avec une couche GRU en utilisant Keras
model = keras.Sequential()
model.add(layers.GRU(128, input_shape=(max_len, len(chars)), return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation="softmax")))

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Entraînement du modèle
model.fit(X, y, batch_size=128, epochs=50)

# Génération de mots de passe
seed_text = "your_seed_text_here"
generated_password = seed_text
for _ in range(50):  # Génère 50 caractères
    x_pred = np.zeros((1, max_len, len(chars)))
    for t, char in enumerate(generated_password):
        x_pred[0, t, char_indices[char]] = 1

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(preds[-1])
    next_char = indices_char[next_index]
    generated_password += next_char

print("Mot de passe généré :", generated_password)
