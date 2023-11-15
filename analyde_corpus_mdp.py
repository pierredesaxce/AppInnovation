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

# Proportion des mots de passe composés de lettres et de chiffres
letters_and_digits_count = sum(word.isalnum() for word in corpus)

# Analyse des combinaisons de caractères
character_pairs = [word[i:i+3] for word in corpus for i in range(len(word)-2)]
character_pair_counts = Counter(character_pairs)

# Détection de mots redondants
unique_corpus = list(set(corpus))

# Affichage des résultats
print(f"Longueur moyenne des mots : {average_length}")
print("Distribution des mots par longueur :")
for length, count in word_length_counts.items():
    print(f"{length} caractères : {count} mots")

print(f"Mots contenant uniquement des lettres : {letters_count} ({round((letters_count/len(corpus))*100,2)} %)")
print(f"Mots contenant uniquement des chiffres : {digits_count} ({round((digits_count/len(corpus))*100,2)} %)")
print(f"Mots contenant des caractères spéciaux : {special_chars_count} ({round((special_chars_count/len(corpus))*100,2)} %)")
print(f"Mots composés de lettres et de chiffres : {letters_and_digits_count} ({round((letters_and_digits_count / len(corpus)) * 100, 2)} %)")

print(f"Nombre de mots uniques : {len(unique_corpus)} ({round((len(unique_corpus)/len(corpus))*100,2)} %)")

# Création d'un graphique pour la distribution des longueurs de mots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(word_length_counts.keys(), word_length_counts.values())
plt.xlabel("Longueur des mots")
plt.ylabel("Nombre de mots")
plt.title("Distribution des longueurs de mots")

# Création d'un graphique pour les combinaisons de caractères les plus fréquentes
plt.subplot(1, 2, 2)
top_pairs = character_pair_counts.most_common(10)
pairs, counts = zip(*top_pairs)
plt.bar(pairs, counts)
plt.xlabel("Combinaisons de caractères")
plt.ylabel("Fréquence")
plt.title("Top 10 des combinaisons de caractères les plus fréquentes")

plt.tight_layout()
plt.show()
