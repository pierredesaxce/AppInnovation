# Générateur de Mots de Passe Inspiré d'Ashley Madison

Ce projet vise à créer un générateur de mots de passe basé sur des modèles d'IA séquentiels, s'inspirant de la structure et des caractéristiques des mots de passe de la liste Ashley Madison. L'approche séquentielle de l'IA permet de capturer les schémas et les structures spécifiques présents dans cette liste tout en générant des mots de passe nouveaux et similaires.

## Utilisation

1. **Installation :**
   - Clonez ce repo sur votre machine locale.

2. **Entraînement du Modèle :**
   - Utilisez des ensembles de données de mots de passe Ashley Madison pour entraîner les modèles d'IA.

3. **Configuration :**
   - Ajustez les paramètres de génération de mots de passe selon vos besoins dans le fichier de configuration.

4. **Génération de Mots de Passe :**
   - Exécutez le générateur pour obtenir des mots de passe inspirés d'Ashley Madison.

## Exemple d'utilisation

```bash
python PasswordGeneratorGlobal.py -te train -m models/model.pt
python PasswordGeneratorGlobal.py -te eval -m models/model.pt
python PasswordGeneratorGlobal.py -te test -m models/model.pt
python PasswordGeneratorGlobal.py -te testf -m models/model.pt
python PasswordGeneratorGlobal.py -te gen -m models/model.pt
