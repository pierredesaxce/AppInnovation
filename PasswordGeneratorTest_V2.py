import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Définir la longueur maximale des mots de passe
max_len = 8  # Remplacez cette valeur par la longueur maximale que vous avez utilisée lors de l'entraînement

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    # Filter passwords with 8 characters
    passwords = [line.strip() for line in file if len(line.strip()) == 8]

# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file if len(line.strip()) == 9]  # Supprimer le dernier caractère "\" à la fin

# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords + test_passwords))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Load the saved model
model = load_model("best_model.h5")  # Load your saved model file here

# Générer des mots de passe en utilisant le modèle entraîné
def generate_passwords(model, seed="", num_passwords=1000000, temperature=3.0):
    print(f"generate_passwords")
    generated_passwords = []

    for i in range(num_passwords):
        generated_password = seed
        input_sequence = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(seed):
            input_sequence[0, t, char_indices[char]] = 1

        for _ in range(8 - len(seed)):
            predictions = model.predict(input_sequence, verbose=0)[0][-1]
            char_index = sample_index(predictions, temperature)
            next_char = indices_char[char_index]
            generated_password += next_char
            input_sequence = np.roll(input_sequence, -1, axis=1)
            input_sequence[0, -1, char_index] = 1

        generated_passwords.append(generated_password)

        print(generated_passwords)
        
        # Imprimer l'avancement à chaque 1000 mots de passe générés
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} passwords")

    return generated_passwords

# Fonction utilitaire pour échantillonner un index en fonction des prédictions du modèle et de la température
def sample_index(predictions, temperature=3.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

# Générer 10000 mots de passe avec une seed spécifique (vous pouvez changer la seed)
generated_passwords = generate_passwords(model, seed="", num_passwords=100)

# Écrire les mots de passe générés dans un fichier txt
with open("generated_passwords.txt", "w") as file:
    for password in generated_passwords:
        file.write(password + "\n")

print("Generated passwords saved to generated_passwords.txt")