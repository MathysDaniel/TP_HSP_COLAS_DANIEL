# Projet de Traitement Parallèle - Utilisation de CUDA

Ce dépôt GitHub contient le code réalisé par **Benjamin COLAS** et **Mathys DANIEL** dans le cadre de plusieurs séances de travaux pratiques (TP) en HSP. L'objectif principal de ces séances était de maîtriser l'utilisation de CUDA, étudier la complexité algorithmique, comparer les performances entre CPU et GPU, implémenter le CNN LeNet-5 pour l'inférence, ainsi que de réaliser diverses opérations matricielles sur GPU.

## Objectifs principaux

### 1. Utilisation de CUDA
Ce projet vise à comprendre et utiliser efficacement CUDA pour le traitement parallèle.

### 2. Analyse de la complexité algorithmique et des performances GPU
L'accent est mis sur l'étude comparative des performances entre CPU et GPU pour évaluer l'accélération obtenue en utilisant CUDA.

### 3. Implémentation de LeNet-5 pour l'inférence
L'objectif final est d'implémenter l'inférence d'un classique CNN, LeNet-5, en se basant sur les concepts décrits dans cet [article](https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture).

## Contenu du Projet

### Opérations sur les matrices avec CUDA

#### Multiplication et Addition de matrices

- Implémentation de fonctions d'opérations matricielles sur GPU (multiplication, addition) en utilisant CUDA.
- Opérations disponibles : allocation de mémoire, création, affichage, addition, multiplication de matrices sur CPU et GPU.
- Les dimensions des matrices peuvent être paramétrées.

#### Analyse de performances

- Estimation de la complexité des opérations matricielles.
- Mesures des temps d'exécution CPU et GPU pour évaluer l'accélération obtenue.
- Comparaison des résultats avec les caractéristiques du GPU utilisé.

## Implémentation des premières couches de LeNet-5

### Génération des données pour Layer 1

#### Layer 1 - Génération des données de test

Dans cette section, nous implémentons les matrices spécifiques nécessaires pour simuler les données d'entrée et de sortie des opérations de la première couche du réseau LeNet-5. Les matrices implémentées sont :

- Une matrice float raw_data de taille 32x32 initialisée avec des valeurs comprises entre 0 et 1, correspondant aux données d'entrée.
- Une matrice float C1_data de taille 6x28x28 initialisée à 0 qui stockera les valeurs de sortie de la convolution 2D (C1) après la première couche.
- Une matrice float S1_data de taille 6x14x14 initialisée à 0 pour stocker les valeurs de sortie du sous-échantillonnage (S1) de la première couche.
- Une matrice float C1_kernel de taille 6x5x5 initialisée avec des valeurs comprises entre 0 et 1, représentant les premiers noyaux de convolution.

Pour créer ces matrices nous effectuons le choix suivant :
- Créer des tableaux à 1 dimension (N=32x32, 6x28x28, 6x14x14 et 6x5x5 respectivement) où chaque case correspond à un élément.

### Convolution 2D

#### Layer 2 - Convolution 2D

- Implémentation de la première convolution 2D avec 6 noyaux de convolution de taille 5x5.
- La taille résultante des données sera de 6x28x28.

### Sous-échantillonnage 2D

#### Layer 3 - Sous-échantillonnage

- Réalisation du sous-échantillonnage 2D par moyennage de 2x2 pixels vers 1 pixel.
- Suite à cette opération, la taille résultante des données sera de 6x14x14.

---

### Fonctions d'activation

#### Fonction d'activation tanh

- Ajout d'une fonction d'activation de type tanh, spécifiquement en sortie de la première convolution.
- Cette fonction d'activation peut être appelée par chaque kernel de votre GPU avec le prototype suivant :

#### Tests après l'ajout de la fonction d'activation

- Après la création de cette fonction d'activation, effectuez à nouveau des tests sur les premières couches pour évaluer les résultats obtenus.

---

L'ajout de cette fonction d'activation, en particulier en sortie de la première convolution, permettra d'améliorer les opérations effectuées dans les premières couches du réseau de neurones LeNet-5.

---

Ce projet a été réalisé dans le cadre des travaux pratiques de HSP pour approfondir la compréhension de CUDA, des opérations matricielles, et pour expérimenter avec LeNet-5 pour l'inférence.
