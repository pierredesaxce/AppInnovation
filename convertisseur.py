# Liste de caractères
chars =  [' ', '!', '#', '$', '%', '&', '(', ')', '*', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '\\', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']

# Tableau passwords in batch
passwords_in_batch = [[65, 69, 69, 66, 69, 77],
 [57, 62, 59, 79, 14, 21],
 [14, 20, 16, 14, 16, 16],
 [70, 76, 15, 15, 18, 21]]



# Convertir les indices en caractères en utilisant la liste chars
converted_passwords = []
for row in passwords_in_batch:
    converted_row = [chars[index] for index in row]
    converted_passwords.append(converted_row)

# Afficher le résultat
for row in converted_passwords:
    print(row)
