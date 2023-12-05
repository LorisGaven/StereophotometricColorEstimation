import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création des données pour la matrice de hauteur z (exemple aléatoire)
# Vous pouvez remplacer cela par votre propre matrice de hauteur
# Assurez-vous que votre matrice z est une matrice 2D contenant les hauteurs pour chaque point
# par exemple, z[i][j] contient la hauteur pour le point (i, j)
# Ici, nous créons une matrice de 10x10 avec des hauteurs aléatoires
z = np.random.rand(10, 10)

# Création des coordonnées x et y pour chaque point de la matrice z
x, y = np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]))

# Création de la figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Affichage de la surface 3D
ax.plot_surface(x, y, z, cmap='viridis')

# Étiquetage des axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Hauteur')

# Affichage
plt.show()