import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    passwords = [line.strip() for line in file]

# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file]  # Supprimer le dernier caractère "\" à la fin

# Création d'un dictionnaire de caractères uniques
chars = sorted(list(set("".join(passwords + test_passwords))))
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}

# Préparation des données d'entraînement
max_len = max([len(password) for password in passwords])
X = np.zeros((len(passwords), max_len, len(chars)), dtype=np.float32)
y = np.zeros((len(passwords), max_len), dtype=np.long)

for i, password in enumerate(passwords):
    for t, char in enumerate(password):
        X[i, t, char_indices[char]] = 1.0
        if t < max_len - 1:
            y[i, t + 1] = char_indices[char]

# Préparation des données de test
max_len_test = max([len(password) for password in test_passwords])
X_test = np.zeros((len(test_passwords), max_len_test, len(chars)), dtype=np.float32)
y_test = np.zeros((len(test_passwords), max_len_test), dtype=np.long)

for i, password in enumerate(test_passwords):
    for t, char in enumerate(password):
        X_test[i, t, char_indices[char]] = 1.0
        if t < max_len_test - 1:
            y_test[i, t + 1] = char_indices[char]

# Création du modèle RNN avec plusieurs couches GRU en utilisant PyTorch
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, 128, batch_first=True)
        self.gru2 = nn.GRU(128, 256, batch_first=True)
        self.gru3 = nn.GRU(256, 512, batch_first=True)
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        out = self.fc(out)
        return out

input_size = len(chars)
hidden_size = 256
output_size = len(chars)

model = GRUModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convertir les tableaux NumPy en torch.Tensor
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Créer des ensembles de données PyTorch
train_dataset = TensorDataset(X_tensor, y_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Créer des chargeurs de données
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Entraînement du modèle
num_epochs = 1
best_test_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, output_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Afficher l'erreur moyenne à chaque epoch
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {average_loss}")

    # Évaluation du modèle sur l'ensemble de test
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct_test_predictions = 0
        total_test_predictions = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, output_size), targets.view(-1))
            test_loss += loss.item()

            # Calcul de l'accuracy sur l'ensemble de test

            _, predicted_index = torch.max(outputs[:, -1, :], 1)
            correct_test_predictions += (predicted_index == targets).sum().item()
            total_test_predictions += targets.numel()

        # Afficher l'erreur moyenne et l'accuracy sur l'ensemble de test
        average_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test_predictions / total_test_predictions
        print(f"Average Test Loss: {average_test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

        # Sauvegarder le meilleur modèle
        if average_test_loss < best_test_loss:
            best_test_loss = average_test_loss
            torch.save(model.state_dict(), "best_model_V2.pt")

print("Training complete.")
