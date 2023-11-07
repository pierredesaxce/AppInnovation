
from collections import Counter
import matplotlib.pyplot as plt

# Nom du fichier contenant le corpus de mots de passe
filename = "Ashley-Madison.txt"

# Lire le contenu du fichier dans une liste
with open(filename, "r") as file:
    corpus = [line.strip() for line in file]

# Calcul de la longueur moyenne des mots
average_length = sum(len(word) for word in corpus) / len(corpus)

# Analyse des mots de différentes longueurs
word_lengths = [len(word) for word in corpus]
word_length_counts = Counter(word_lengths)

# Identification des caractères utilisés
letters_count = sum(word.isalpha() for word in corpus)
digits_count = sum(word.isdigit() for word in corpus)
special_chars_count = sum(not word.isalnum() for word in corpus)

# Détection de mots redondants
unique_corpus = list(set(corpus))

# Affichage des résultats
print(f"Longueur moyenne des mots : {average_length}")
print("Distribution des mots par longueur :")
for length, count in word_length_counts.items():
    print(f"{length} caractères : {count} mots")

print(f"Mots contenant uniquement des lettres : {letters_count}")
print(f"Mots contenant des chiffres : {digits_count}")
print(f"Mots contenant des caractères spéciaux : {special_chars_count}")

print(f"Nombre de mots uniques : {len(unique_corpus)}")

# Création d'un graphique pour la distribution des longueurs de mots
plt.bar(word_length_counts.keys(), word_length_counts.values())
plt.xlabel("Longueur des mots")
plt.ylabel("Nombre de mots")
plt.show()
