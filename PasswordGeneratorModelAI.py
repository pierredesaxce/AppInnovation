import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Limit GPU memory usage
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # Limit GPU memory fraction
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # Adjust the limit as needed
        )

        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU.")

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

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle RNN avec une couche GRU en utilisant Keras
model = keras.Sequential()
model.add(layers.GRU(128, input_shape=(max_len, len(chars)), return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation="softmax")))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  # Ajout de 'accuracy' comme métrique

# Define a ModelCheckpoint callback
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)  # Changement du monitor

# Entraînement du modèle sur l'ensemble d'entraînement
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=1, callbacks=[checkpoint])

# Afficher l'accuracy sur les ensembles d'entraînement et de test
train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

print(f"Accuracy on training set: {train_accuracy[-1]}")
print(f"Accuracy on test set: {test_accuracy[-1]}")