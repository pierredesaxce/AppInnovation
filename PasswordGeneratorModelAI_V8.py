import os
import sys
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
    # Filter passwords with max_len characters
    passwords = [line.strip() for line in file if len(line.strip()) == max_len]

# Afficher la taille de la liste passwords
print("Taille de passwords:", len(passwords))

# Diviser les données en ensembles d'entraînement et de validation
train_passwords, val_passwords = train_test_split(passwords, test_size=0.2, random_state=42)


class CustomAccuracy(Metric):
    def __init__(self, name="custom_accuracy", **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.total_positions = self.add_weight(name="total_positions", initializer="zeros")
        self.correct_positions = self.add_weight(name="correct_positions", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot encoded vectors to indices
        y_true_indices = tf.argmax(y_true, axis=-1)
        y_pred_indices = tf.argmax(y_pred, axis=-1)

        # Print generated passwords for each batch
        #tf.print("Generated passwords in batch:", y_pred_indices, output_stream=sys.stdout, summarize=-1, end="\n")

        #  passwords for each batch
        #tf.print("passwords in batch:", y_true_indices, output_stream=sys.stdout, summarize=-1, end="\n")


        #tf.print("y_true_indices : ", y_true_indices)
        #tf.print("y_pred_indices : ", y_pred_indices)

        # Compare if generated passwords match test passwords at each position
        correct_positions = tf.cast(tf.math.equal(y_true_indices, y_pred_indices), tf.float32)

        # Update counts
        self.total_positions.assign_add(tf.cast(tf.size(y_true_indices), tf.float32))
        self.correct_positions.assign_add(tf.reduce_sum(correct_positions))

        #tf.print("total_positions : ", self.total_positions)
        #tf.print("correct_positions : ", self.correct_positions)

    def result(self):
        return self.correct_positions / self.total_positions





# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Préparation des données d'entraînement
max_len_train = max([len(password) for password in train_passwords])
X_train = np.zeros((len(train_passwords), max_len_train, len(chars)), dtype=np.float32)
y_train = np.zeros((len(train_passwords), max_len_train, len(chars)), dtype=np.float32)

for i, password in enumerate(train_passwords):
    for t, char in enumerate(password):
        X_train[i, t, char_indices[char]] = 1
        if t < max_len_train:
            y_train[i, t, char_indices[char]] = 1

# Préparation des données de validation
max_len_val = max([len(password) for password in val_passwords])
X_val = np.zeros((len(val_passwords), max_len_val, len(chars)), dtype=np.float32)
y_val = np.zeros((len(val_passwords), max_len_val, len(chars)), dtype=np.float32)

for i, password in enumerate(val_passwords):
    for t, char in enumerate(password):
        X_val[i, t, char_indices[char]] = 1
        if t < max_len_val:
            y_val[i, t, char_indices[char]] = 1

# Création du modèle RNN avec plusieurs couches GRU
model = keras.Sequential()
model.add(layers.LSTM(16, input_shape=(max_len, len(chars)), return_sequences=True))
model.add(layers.LSTM(32, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation="softmax")))

# Create an instance of the custom accuracy metric
custom_accuracy = CustomAccuracy()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[custom_accuracy])

# Define a ModelCheckpoint callback
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_custom_accuracy', save_best_only=True, mode='max', verbose=1)

# Entraînement du modèle sur l'ensemble d'entraînement
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=10, callbacks=[checkpoint])

# Afficher l'accuracy sur les ensembles d'entraînement et de test à chaque epoch
for epoch in range(len(history.history['custom_accuracy'])):
    train_accuracy = history.history['custom_accuracy'][epoch]
    val_accuracy = history.history['val_custom_accuracy'][epoch]
    print(f"Epoch {epoch + 1}/{len(history.history['custom_accuracy'])} - Accuracy on training set: {train_accuracy} - Accuracy on validation set: {val_accuracy}")
