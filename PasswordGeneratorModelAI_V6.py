import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Metric

#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Définir la longueur maximale des mots de passe
max_len = 6  # Remplacez cette valeur par la longueur maximale que vous avez utilisée lors de l'entraînement

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    # Filter passwords with 8 characters
    passwords = [line.strip() for line in file if len(line.strip()) == max_len]

# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file if len(line.strip()) == max_len+1]  # Supprimer le dernier caractère "\" à la fin


class CustomAccuracy(Metric):
    def __init__(self, name="custom_accuracy", **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")
        self.correct_predictions = self.add_weight(name="correct_predictions", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot encoded vectors to indices
        y_true_indices = tf.argmax(y_true, axis=-1)
        y_pred_indices = tf.argmax(y_pred, axis=-1)

        # Compare if generated passwords match test passwords
        correct = tf.reduce_all(tf.math.equal(y_true_indices, y_pred_indices), axis=-1)
        correct = tf.cast(correct, tf.float32)

        # Update counts
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.correct_predictions.assign_add(tf.reduce_sum(correct))

    def result(self):
        return self.correct_predictions / self.total_samples

# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords + test_passwords))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Préparation des données d'entraînement
max_len = max([len(password) for password in passwords])
X = np.zeros((len(passwords), max_len, len(chars)), dtype=np.float32)
y = np.zeros((len(passwords), max_len, len(chars)), dtype=np.float32)

for i, password in enumerate(passwords):
    for t, char in enumerate(password):
        X[i, t, char_indices[char]] = 1
        if t < max_len - 1:
            y[i, t + 1, char_indices[char]] = 1

# Préparation des données de test
max_len_test = max([len(password) for password in test_passwords])
X_test = np.zeros((len(test_passwords), max_len_test, len(chars)), dtype=np.float32)
y_test = np.zeros((len(test_passwords), max_len_test, len(chars)), dtype=np.float32)

for i, password in enumerate(test_passwords):
    for t, char in enumerate(password):
        X_test[i, t, char_indices[char]] = 1
        if t < max_len_test - 1:
            y_test[i, t + 1, char_indices[char]] = 1

# Création du modèle RNN avec plusieurs couches GRU
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(max_len, len(chars)), return_sequences=True))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(256, return_sequences=True))
model.add(layers.LSTM(512, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation="softmax")))

# Create an instance of the custom accuracy metric
custom_accuracy = CustomAccuracy()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[custom_accuracy])

# Define a ModelCheckpoint callback
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_custom_accuracy', save_best_only=True, mode='max', verbose=1)

# Entraînement du modèle sur l'ensemble d'entraînement
history = model.fit(X, y, validation_data=(X_test, y_test), batch_size=4096, epochs=100, callbacks=[checkpoint])

# Afficher l'accuracy sur les ensembles d'entraînement et de test à chaque epoch
for epoch in range(len(history.history['custom_accuracy'])):
    train_accuracy = history.history['custom_accuracy'][epoch]
    test_accuracy = history.history['val_custom_accuracy'][epoch]
    print(f"Epoch {epoch + 1}/{len(history.history['custom_accuracy'])} - Accuracy on training set: {train_accuracy} - Accuracy on test set: {test_accuracy}")