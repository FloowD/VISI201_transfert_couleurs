#!/usr/bin/env python

from PIL import Image
from random import randint
from operator import itemgetter
import time
import numpy as np
from numpy import*

best_u = (0,0,0)
quality = 0
matrice_passage = np.array([[0.618,0.177,0.205],
                           [0.299,0.587,0.114],
                           [0,0.056,0.944]],dtype=np.float64)

matrice_passage2 = np.array([[1.816,-0.533,-0.343],
                           [-0.967,1.998,-0.031],
                           [0.057,-0.118,1.061]],dtype=np.float64)

""" Légende :
    im : image
    _RGB : objet image en rgb
    _rgb : liste de tuples rgb
    _ord : _rgb ordonné selon u
    u : vecteur tiré aléatoirement
    cost : coût
    _cur : courant """

def intro(fichier1, fichier2):
    """ Entrées : deux fichiers image
        Sortie : booléen
        Vérifie la compatibilité de taille et mode des deux images """

    im1 = Image.open(fichier1)
    im2 = Image.open(fichier2)

    print()
    print("=== Image A ===\nFormat : {}\nTaille : {}\nMode : {}".format(im1.format, im1.size, im1.mode))
    print()
    print("=== Image B ===\nFormat : {}\nTaille : {}\nMode : {}".format(im2.format, im2.size, im2.mode))
    print()
    if im1.size==im2.size and im1.mode==im2.mode :
        print("Fichiers OK")
        print()
        res = True
    else :
        print("ATTENTION, incompatibilité des fichiers")
        res = False


    im1.close()
    im2.close()
    return res


####################
#   Partie RGB -> XYZ
def to_rgb_numpy(im):
    """ Entrée : image
        Sortie : liste de tuples
        Convertie un objet image en liste rgb """

    im_rgb = np.array(im)
    #De base le tableau des codes RGB est en 3 dimensions, pour utiliser au mieux la version précédente
    #je transforme ce tableau en 2 dimensiosn
    im_true = im_rgb.reshape((-1,3))


    return(im_true)

def RGB_to_XYZ(tab_RGB):
    """On effectue le produit matriciel entre les composantes RGB et la matrice de passage"""
    global matrice_passage
    tab_transfert = np.dot(matrice_passage,tab_RGB)

    return tab_transfert

def to_XYZ(im):
    """Entrée: image
       Sortie : un tableau a 2 dimensions
       On transforme l'image dans l'espace XYZ et on sotck les valeurs dans un tableau"""

    tab_RGB = to_rgb_numpy(im)
    tabXYZ = np.apply_along_axis(RGB_to_XYZ, 1, tab_RGB)

    return tabXYZ
        

#   Partie RGB -> XYZ
#####################

####################
#   Partie XYZ -> Lab
def func_transi(tab,i):
    """tab -> tab des valeurs
        i ->indice X:0 Y:1 Z:2
        Fait la transition pour appliquer la fonction 'f' sur X,Y,Z"""
    
    return tab[i:i+1]

def f(t):
    """Pour le passage XYZ -> Lab"""
    if t > (6/29)**3:
        valeur = t**(1/3)
    else:
        valeur = (1/3)*((29/6)**2)*t+(4/29)
    
    return valeur

def to_Lab(tab_XYZ):
    """Entrée : Tableau dans la base XYZ
       Sortie : Un tableau dans la base Lab"""
    Xn = 255
    Yn = 255
    Zn = 255
    tab_Lab = np.zeros((len(tab_XYZ),3))

    tab_X = np.apply_along_axis(func_transi, 1, tab_XYZ,0)
    tab_Y = np.apply_along_axis(func_transi, 1, tab_XYZ,1)
    tab_Z = np.apply_along_axis(func_transi, 1, tab_XYZ,2)

    
    tab_X_Xn = np.apply_along_axis(f, 1, tab_X/Xn)
    tab_Y_Yn = np.apply_along_axis(f, 1, tab_Y/Yn)
    tab_Z_Zn = np.apply_along_axis(f, 1, tab_Z/Zn)

    
    tab_L = 116*tab_Y_Yn - 16
    tab_a = 500*(tab_X_Xn - tab_Y_Yn)
    tab_b = 200*(tab_Y_Yn - tab_Z_Zn)

    tab_L = tab_L.reshape(len(tab_L),)
    tab_a = tab_a.reshape(len(tab_a),)
    tab_b = tab_b.reshape(len(tab_b),)
    
    tab_Lab[:,0] = tab_L
    tab_Lab[:,1] = tab_a
    tab_Lab[:,2] = tab_b

    return (tab_Lab)

#   Partie XYZ -> Lab
#####################

#################
# Lab -> XYZ
def f2(t):
    """Pour le passage XYZ -> Lab"""
    if t > (6/29):
        valeur = t**3
    else:
        valeur = 3*((6/29)**2)*(t-(4/29))
    
    return valeur

def from_Lab_to_XYZ(tab_Lab):
    """Entrée tab Lab
       Sortie : tab XYZ"""
    Xn = 255
    Yn = 255
    Zn = 255
    tab_XYZ = np.zeros((len(tab_Lab),3))

    tab_L = np.apply_along_axis(func_transi, 1, tab_Lab,0)
    tab_a = np.apply_along_axis(func_transi, 1, tab_Lab,1)
    tab_b = np.apply_along_axis(func_transi, 1, tab_Lab,2)
    
    tab_Y = np.apply_along_axis(f2, 1, (tab_L+16)/116) * Yn
    tab_X = np.apply_along_axis(f2, 1, (tab_L+16)/116 + (tab_a/500)) * Xn
    tab_Z = np.apply_along_axis(f2, 1, (tab_L+16)/116 - (tab_b/500)) * Zn

    tab_X = tab_X.reshape(len(tab_X),)
    tab_Y = tab_Y.reshape(len(tab_Y),)
    tab_Z = tab_Z.reshape(len(tab_Z),)

    tab_XYZ[:,0] = tab_X
    tab_XYZ[:,1] = tab_Y
    tab_XYZ[:,2] = tab_Z

    return tab_XYZ


# Lab -> XYZ
#################

#################
# XYZ -> RGB

def from_XYZ_to_RGB(tab_XYZ):
    """Entrée: tab XYZ
       Sortie: tab RGB"""
    global matrice_passage2
    tab_RGB = np.apply_along_axis(calc, 1, tab_XYZ)

    return tab_RGB
    
def calc(tab_XYZ):
    global matrice_passage2
    tab_RGB = np.dot(matrice_passage2, tab_XYZ)

    return tab_RGB


# XYZ -> Lab
#################

def to_ord_numpy(tab1_Lab, tab2_Lab, u):
    """ Entrée : deux listes de tuples et un tuple (vecteur)
        Sortie : un tuple de deux listes de tuples
        Ordonne deux listes Lab selon un vecteur u """

    im1_ord = np.zeros((len(tab1_Lab),5),dtype=np.int64)
    im2_ord = np.zeros((len(tab1_Lab),5),dtype=np.int64)

    v1_scalaire = tab1_Lab*u
    v2_scalaire = tab2_Lab*u

    tab1_somme = np.sum(v1_scalaire, axis=1)
    tab2_somme = np.sum(v2_scalaire, axis=1)

    im1_ord[:,0] = tab1_somme
    im2_ord[:,0] = tab2_somme

    ################
    #Stock code RGB
    im1_ord[:,1:4] = tab1_Lab
    im2_ord[:,1:4] = tab2_Lab

    #################
    #Stock les indices
    #Je ne crée qu'un seul tableau d'indice car les images sont de même taille
    index = np.arange(len(tab1_Lab))
    
    im1_ord[:,4] = index
    im2_ord[:,4] = index
    

    #Trier les 2 listes
    im1_ord = im1_ord[im1_ord[:,0].argsort()]
    im2_ord = im2_ord[im2_ord[:,0].argsort()]

    #Stocker les indices de l'image 1 dans l'image 2
    im2_ord[:,4:] = im1_ord[:,4:]

    
    return(im1_ord, im2_ord)


def cost_Lab(tab1_Lab, tab2_Lab, u):
    """ Entrée : deux listes de tuples et un tuple
        Sortie : un entier
        Calcul le coût entre deux listes rgb selon un vecteur u """

    im12_ord = to_ord_numpy(tab1_Lab, tab2_Lab, u)
    im1_ord = im12_ord[0]
    im2_ord = im12_ord[1]
    ecart2 = 0.0

    tab1_Lab = im1_ord[:,0]
    tab2_Lab = im2_ord[:,0]

    ecart = sqrt((tab2_Lab - tab1_Lab)**2)
    ecart2 = ecart.sum()

    return ecart2



def best_cost_numpy(im1_rgb, im2_rgb):
    """ Entrée : deux listes de tuples
        Sortie : un entier
        Calcul le meilleur coût entre deux listes rgb """

    global best_u

    u = (randint(-500,500), randint(-500,500), randint(-500,500)) # rand en flottant !
    best_cost = cost_Lab(im1_rgb, im2_rgb, u)


    nb_u = 20 # nombre de tirages de u
    cost_list = [] # liste de tout les coûts que l'on va calculer
    for i in range(0, nb_u):
        u = (randint(-500,500), randint(-500,500), randint(-500,500))
        cost_cur = cost_Lab(im1_rgb, im2_rgb, u)
        cost_list += [ cost_cur ]

        # echo
        print( i+1, best_cost, cost_cur )

        if cost_cur < best_cost :
            best_cost = cost_cur
            best_u = u

    quality = min(cost_list) / max(cost_list)

    print()
    print("Meilleur vecteur u :", best_u)
    print("Qualité :", quality)
    print()
    return best_cost



def new_image_numpy(fichier1, fichier2, fichier3):
    """ Entrées : deux fichiers image, puis le nom du fichier créé
        Sortie : rien
        Créer un fichier en transférant la palette du fichier 2 vers le fichier 1 """

    global best_u

    im1 = Image.open(fichier1)
    im2 = Image.open(fichier2)
    
    print("Transfert dans l'espace XYZ...")
    im11_XYZ = to_XYZ(im1)
    im22_XYZ = to_XYZ(im2)

    print("Transfert dans l'espace L*a*b...")
    im1_rgb = to_Lab(im11_XYZ)
    im2_rgb = to_Lab(im22_XYZ)

    best_cost_numpy(im1_rgb, im2_rgb)

    # on récuperre les deux listes ordonnées par le meilleur vercteur
    var = to_ord_numpy(im1_rgb, im2_rgb, best_u)
    im1_ord = var[0]
    im2_ord = var[1]
    im3_rgb = [0]*len(im1_ord)

    
    im2_ord = im2_ord[im2_ord[:,4].argsort()]

    im3_Lab = im2_ord[:,1:4]
    print("Transfert dans l'espace XYZ...")
    im3_XYZ = from_Lab_to_XYZ(im3_Lab)
    print("Transfert dans l'espace RGB...")
    im3_RGB = from_XYZ_to_RGB(im3_XYZ)
    im3_RGB = im3_RGB.astype(int)
    
    im3 = Image.new(im1.mode, im1.size)
    #On transforme le tableau en tuple afin de pouvoir l'enregistrer avec im.putdata
    im3_bis = tuple(map(tuple, im3_RGB))
    im3.putdata(im3_bis)
    
    print()
    print("sauvegarde du nouveau fichier")
    im3.save(fichier3, 'BMP')
    print("Fin")

    im1.close()
    im2.close()


def main_numpy(fichier1, fichier2, fichier3):
    if intro(fichier1, fichier2):
        new_image_numpy(fichier1, fichier2, fichier3)

