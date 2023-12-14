import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.metrics import Metric


# Définir la longueur maximale des mots de passe
max_len = 6  # Remplacez cette valeur par la longueur maximale que vous avez utilisée lors de l'entraînement

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    # Filter passwords with 8 characters
    passwords = [line.strip() for line in file if len(line.strip()) == max_len]

# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    # Filter passwords with 9 characters
    test_passwords = [line.strip()[:-1] for line in file if len(line.strip()) == max_len + 1]  # Supprimer le dernier caractère "\" à la fin

# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords + test_passwords))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

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

# Register the custom object
with custom_object_scope({'CustomAccuracy': CustomAccuracy}):
    model = load_model("best_model.h5")

# Générer des mots de passe en utilisant le modèle entraîné
def generate_passwords(model, seed="", num_passwords=1000000, temperature=3.0):
    generated_passwords = []

    for i in range(num_passwords):
        generated_password = seed
        input_sequence = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(seed):
            input_sequence[0, t, char_indices[char]] = 1

        for _ in range(max_len - len(seed)):
            predictions = model.predict(input_sequence, verbose=0)[0][-1]
            char_index = sample_index(predictions, temperature)
            next_char = indices_char[char_index]
            generated_password += next_char
            input_sequence = np.roll(input_sequence, -1, axis=1)
            input_sequence[0, -1, char_index] = 1

        generated_passwords.append(generated_password)

        # Calculer le pourcentage de lettres correspondant aux max_len derniers caractères du mot de passe original
        matching_percentage = sum(c1 == c2 for c1, c2 in zip(seed_password[-int(max_len/2):], generated_password[-int(max_len/2):])) / int(max_len/2) * 100


        # Imprimer chaque mot de passe sur une ligne avec le pourcentage correspondant
        print(f"Original Password: {seed_password} - Generated Password: {generated_password} - Match: {seed_password == generated_password} - Matching Percentage: {matching_percentage:.2f}%")
        
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

# Générer 10000 mots de passe avec les max_len premiers caractères de chaque mot de passe dans eval.txt comme seed
generated_passwords = []
match_count = 0
total_matching_percentage = 0

for seed_password in test_passwords:
    seed = seed_password[:int(max_len/2)]
    generated_password = generate_passwords(model, seed=seed, num_passwords=1)[0]
    generated_passwords.append(generated_password)
    
    # Calculer le pourcentage de lettres correspondant au mot de passe original
    matching_percentage = sum(c1 == c2 for c1, c2 in zip(seed_password[-int(max_len/2):], generated_password[-int(max_len/2):])) / int(max_len/2) * 100
    total_matching_percentage += matching_percentage
    
    # Vérifier si le mot de passe généré correspond au mot de passe attendu
    if seed_password == generated_password:
        match_count += 1

# Écrire les mots de passe générés dans un fichier txt
with open("generated_passwords.txt", "w") as file:
    for password in generated_passwords:
        file.write(password + "\n")

# Calculer les statistiques finales
total_generated = len(generated_passwords)
accuracy = match_count / total_generated if total_generated > 0 else 0.0
average_matching_percentage = total_matching_percentage / total_generated if total_generated > 0 else 0.0

# Afficher les statistiques finales
print(f"Total passwords generated: {total_generated}")
print(f"Matching passwords: {match_count}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Matching Percentage: {average_matching_percentage:.2f}%")
print("Generated passwords saved to generated_passwords.txt")
