# Liste de caractères
chars =  [' ', '!', '#', '$', '%', '&', '(', ')', '*', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '\\', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']

# Tableau passwords in batch
passwords_in_batch = [
  [19, 13, 15, 17, 17, 16],
  [73, 70, 59, 59, 72, 73],
  [67, 55, 57, 21, 21, 21],
  [55, 76, 69, 68, 12, 13],
  [61, 69, 72, 67, 55, 68]
]

# Convertir les indices en caractères en utilisant la liste chars
converted_passwords = []
for row in passwords_in_batch:
    converted_row = [chars[index] for index in row]
    converted_passwords.append(converted_row)

# Afficher le résultat
for row in converted_passwords:
    print(row)
