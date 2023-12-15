import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    # Filter passwords with max_len characters
    passwords = [line.strip() for line in file if len(line.strip()) == max_len]


# Chargement du fichier eval.txt pour les données de test
with open("data/eval.txt", "r") as file:
    test_passwords = [line.strip()[:-1] for line in file]  # Supprimer le dernier caractère "\" à la fin

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

# Conversion en torch.Tensor
X = torch.from_numpy(X)
y = torch.from_numpy(y)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# Création du modèle RNN avec plusieurs couches GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru3 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru4 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        out, _ = self.gru4(out)
        out = self.fc(out)
        return out

model = GRUModel(len(chars), 128, len(chars))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Entraînement du modèle sur l'ensemble d'entraînement
batch_size = 128
num_epochs = 1

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.permute(0, 2, 1), labels.argmax(dim=2))
        loss.backward()
        optimizer.step()

# Afficher l'accuracy sur les ensembles d'entraînement et de test à chaque epoch
with torch.no_grad():
    model.eval()
    train_accuracy = (model(X).argmax(dim=2) == y.argmax(dim=2)).float().mean().item()
    test_accuracy = (model(X_test).argmax(dim=2) == y_test.argmax(dim=2)).float().mean().item()
    print(f"Accuracy on training set: {train_accuracy} - Accuracy on test set: {test_accuracy}")

    # Sauvegarder le meilleur modèle
    if average_test_loss < best_test_loss:
        best_test_loss = average_test_loss
        torch.save(model.state_dict(), "best_model_V4.pt")
