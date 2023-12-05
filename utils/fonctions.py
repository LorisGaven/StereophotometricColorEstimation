import cv2
from scipy.signal import convolve2d
import numpy as np
from tqdm import tqdm

def lightDirection(boule, rayon):
    # On suppose que boule est l'image avec seulement la boule et qu'elle est centrée
    taille_noyau = round(rayon / 20)

    # Créer le noyau triangle (filtre de moyenne)
    noyau_triangle = np.tri(taille_noyau, taille_noyau)
    noyau_triangle /= noyau_triangle.sum()

    # Appliquer le filtre de convolution
    boule_filtree = convolve2d(boule, noyau_triangle, mode='same', boundary='symm')

    hauteur, largeur = boule_filtree.shape
    centre = (largeur // 2, hauteur // 2)

    # Créer un masque noir de la même taille que l'image
    masque = np.zeros((hauteur, largeur), dtype=np.uint8)
    # Dessiner un cercle blanc sur le masque
    cv2.circle(masque, centre, int(rayon * 0.9), (255), thickness=-1)  # Épaisseur négative pour remplir le cercle

    boule_masque = cv2.bitwise_and(boule_filtree, boule_filtree, mask=masque)

    _, light, _ , coord_light = cv2.minMaxLoc(boule_masque)
    a,b = coord_light

    x = b - rayon
    y = rayon - a

    z = np.sqrt(rayon**2 - x**2 - y**2)
    n = np.array([x, y, z]) / rayon

    v = np.array([0, 0, 1])

    l = 2 * np.dot(n,v) * n - v

    return l.tolist()

def lightsDirection(images, boule_position):
    a,b,c,d = boule_position

    ls = []
    for image in tqdm(images):
        boule = image[a:b, c:d, :]
        boule = cv2.cvtColor(boule, cv2.COLOR_BGR2GRAY)
        rayon = (b - a) / 2

        l = lightDirection(boule, rayon)
        ls.append(l)

    return np.array(ls)

def getI(images):
    nb_image = len(images)
    nb_pixel =  images[0].shape[0] * images[0].shape[1]

    I = np.zeros([nb_image, nb_pixel])

    for i in range(nb_image):
        image = images[i]
        I[i,:] = image.flatten()

    return I

def integrationSCS(p, q, gt=None):
    # Compute div(p,q)
    px = 0.5 * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
    qy = 0.5 * (np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1))

    # Div(p,q) + Boundary Condition
    f = px + qy
    f[0, 1:-1] = 0.5 * (p[0, 1:-1] + p[1, 1:-1])
    f[-1, 1:-1] = 0.5 * (-p[-1, 1:-1] - p[-2, 1:-1])
    f[1:-1, 0] = 0.5 * (q[1:-1, 0] + q[1:-1, 1])
    f[1:-1, -1] = 0.5 * (-q[1:-1, -1] - q[1:-1, -2])

    f[0, 0] = 0.5 * (p[0, 0] + p[1, 0] + q[0, 0] + q[0, 1])
    f[-1, 0] = 0.5 * (-p[-1, 0] - p[-2, 0] + q[-1, 0] + q[-1, 1])
    f[0, -1] = 0.5 * (p[0, -1] + p[1, -1] - q[0, -1] - q[0, -2])
    f[-1, -1] = 0.5 * (-p[-1, -1] - p[-2, -1] - q[-1, -1] - q[-1, -2])

    # Cosine Transform of f
    fsin = dctn(f, norm='ortho')

    # Denominator
    x, y = np.meshgrid(np.arange(p.shape[1]), np.arange(p.shape[0]))
    denom = (2 * np.cos(np.pi * x / p.shape[1]) - 2) + (2 * np.cos(np.pi * y / p.shape[0]) - 2)
    Z = fsin / denom
    Z[0, 0] = 0.5 * (Z[0, 1] + Z[1, 0])  # Or whatever...

    # Inverse Cosine Transform
    U = idctn(Z, norm='ortho')

    if gt is not None:
        moyenne_ecarts = np.mean(U - gt)
        U -= moyenne_ecarts
        npix = p.shape[0] * p.shape[1]
        rmse = np.sqrt(np.sum((U - gt) ** 2) / npix)
        return U, rmse
    else:
        U -= np.min(U)
        return U