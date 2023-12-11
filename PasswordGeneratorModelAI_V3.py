import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    passwords = [line.strip() for line in file]

# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file]  # Supprimer le dernier caractère "\" à la fin

# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords + test_passwords))))
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

# Préparation des données de test
max_len_test = max([len(password) for password in test_passwords])
X_test = np.zeros((len(test_passwords), max_len_test, len(chars)), dtype=np.bool_)
y_test = np.zeros((len(test_passwords), max_len_test, len(chars)), dtype=np.bool_)

for i, password in enumerate(test_passwords):
    for t, char in enumerate(password):
        X_test[i, t, char_indices[char]] = 1
        if t < max_len_test - 1:
            y_test[i, t + 1, char_indices[char]] = 1

# Création du modèle RNN avec plusieurs couches GRU
model = keras.Sequential()
model.add(layers.GRU(128, input_shape=(max_len, len(chars)), return_sequences=True))
model.add(layers.GRU(64, return_sequences=True))
model.add(layers.GRU(32, return_sequences=True))
model.add(layers.GRU(64, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation="softmax")))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Define a ModelCheckpoint callback
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Entraînement du modèle sur l'ensemble d'entraînement
history = model.fit(X, y, validation_data=(X_test, y_test), batch_size=512, epochs=10, callbacks=[checkpoint])


# Afficher l'accuracy sur les ensembles d'entraînement et de test à chaque epoch
for epoch in range(len(history.history['accuracy'])):
    train_accuracy = history.history['accuracy'][epoch]
    test_accuracy = history.history['val_accuracy'][epoch]
    print(f"Epoch {epoch + 1}/{len(history.history['accuracy'])} - Accuracy on training set: {train_accuracy} - Accuracy on test set: {test_accuracy}")
