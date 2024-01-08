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

# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file]  # Supprimer le dernier caractère "\" à la fin

# Définir les tailles d'entrée, de sortie et cachée
input_size = len(chars)
hidden_size = 32
output_size = len(chars)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, 128, batch_first=True)
        self.gru2 = nn.GRU(128, 256, batch_first=True)
        self.gru3 = nn.GRU(256, 128, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        out = self.fc(out)
        return out

# Charger le meilleur modèle
best_model = GRUModel(input_size, hidden_size, output_size).to(device)
best_model.load_state_dict(torch.load("best_model_V2.pt", map_location=device))
best_model.eval()

# Fonction pour générer des mots de passe à partir du modèle
def generate_password(model, seed="", max_len=20):
    generated_password = seed
    input_sequence = torch.zeros((1, len(seed), len(chars)), dtype=torch.float32).to(device)

    for t, char in enumerate(seed):
        input_sequence[0, t, char_indices[char]] = 1.0

    for _ in range(max_len - len(seed)):
        outputs = model(input_sequence)
        _, predicted_index = torch.max(outputs, 2)
        next_char = indices_char[predicted_index.item()]
        generated_password += next_char

        input_sequence = torch.roll(input_sequence, -1, dims=1)
        input_sequence[0, -1, char_indices[next_char]] = 1.0

    return generated_password

# Générer quelques mots de passe avec une seed spécifique
for seed in ["abc", "123", "xyz"]:
    generated_password = generate_password(best_model, seed=seed, max_len=15)
    print(f"Seed: {seed}, Generated Password: {generated_password}")
