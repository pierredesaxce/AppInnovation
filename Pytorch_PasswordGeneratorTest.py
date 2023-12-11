# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    passwords = [line.strip() for line in file]

# Chargement du fichier eval txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file]  # Supprimer le dernier caractère "\" à la fin


# Chargement du fichier eval txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords_seed = [line.strip()[:-4] for line in file]  # Supprimer le dernier caractère "\" à la fin

chars = sorted(list(set("".join(passwords + test_passwords))))
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}


# Définir les tailles d'entrée, de sortie et cachée
input_size = len(chars)
hidden_size = 32
output_size = len(chars)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru3 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        out = self.fc(out)
        return out

# Charger le meilleur modèle
best_model = GRUModel(input_size, hidden_size, output_size).to(device)
best_model.load_state_dict(torch.load("best_model.pt", map_location=device))
best_model.eval()

def generate_password(model, seed="", max_len=20, temperature=1.0):
    generated_password = seed
    input_sequence = torch.zeros((1, len(seed), len(chars)), dtype=torch.float32).to(device)

    for t, char in enumerate(seed):
        input_sequence[0, t, char_indices[char]] = 1.0

    for _ in range(max_len - len(seed)):
        outputs = model(input_sequence)

        probabilities = torch.softmax(outputs / temperature, dim=2)
        predicted_index = torch.multinomial(probabilities[0, -1, :], 1)
        next_char = indices_char[predicted_index.item()]
        generated_password += next_char

        input_sequence = torch.zeros((1, len(generated_password), len(chars)), dtype=torch.float32).to(device)
        for t, char in enumerate(generated_password):
            input_sequence[0, t, char_indices[char]] = 1.0

    return generated_password



def test_generated_passwords(model, test_passwords_seed, num_tests=1000, max_len=13, temperature=1.0):
    correct_count = 0

    for seed in test_passwords_seed[:num_tests]:
        generated_password = generate_password(model, seed=seed, max_len=max_len, temperature=temperature)
        print("Seed: {}, Generated Password: {}".format(seed, generated_password))

        # Vérifier si le mot de passe généré correspond à l'un des mots de passe de test
        if generated_password in test_passwords_seed:
            print("Correct! Generated password matches a test password.")
            correct_count += 1
        else:
            print("Incorrect. Generated password does not match any test password.")
        
        print("\n" + "="*30 + "\n")

    accuracy = correct_count / num_tests
    print("Accuracy: {:.2%}".format(accuracy))

# Utilisation de la fonction pour tester les mots de passe générés
test_generated_passwords(best_model, test_passwords_seed)

