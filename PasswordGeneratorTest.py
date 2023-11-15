import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Définir la longueur maximale des mots de passe
max_len = 110  # Remplacez cette valeur par la longueur maximale que vous avez utilisée lors de l'entraînement

# Définir la liste de caractères uniques (chiffres, lettres et caractères spéciaux)
chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"?$%&\'()*+,-./<=>?@[\\]^_`{|}~'

# Load the saved model
model = load_model("best_model.h5")  # Load your saved model file here

# Génération de mots de passe
seed_text = "abcde"
generated_password = seed_text
for _ in range(7):  # Génère 10 caractères
    x_pred = np.zeros((1, max_len, len(chars)))
    for t, char in enumerate(generated_password):
        x_pred[0, t, chars.index(char)] = 1

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(preds[-1])
    next_char = chars[next_index]
    generated_password += next_char

print("Mot de passe généré :", generated_password)
