#!/usr/bin/env python

from PIL import Image
from random import randint
from operator import itemgetter
import time
import numpy as np
from numpy import*

best_u = (0,0,0)
quality = 0

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

    global longueur
    global largeur

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


def to_rgb_numpy(im):
    """ Entrée : image
        Sortie : liste de tuples
        Convertie un objet image en liste rgb """

    im_rgb = np.array(im)
    #De base le tableau des codes RGB est en 3 dimensions, pour utiliser au mieux la version précédente
    #je transforme ce tableau en 2 dimensiosn
    im_true = im_rgb.reshape((-1,3))


    return(im_true)



def to_ord_numpy(im1_rgb, im2_rgb, u):
    """ Entrée : deux listes de tuples et un tuple (vecteur)
        Sortie : un tuple de deux listes de tuples
        Ordonne deux listes rgb selon un vecteur u """
         

    im1_ord = np.zeros((len(im1_rgb),5),dtype=np.int64)
    im2_ord = np.zeros((len(im1_rgb),5),dtype=np.int64)

    ###############
    #Stock RGB*u

    v1_scalaire = im1_rgb*u
    v2_scalaire = im2_rgb*u

    tab1_somme = np.sum(v1_scalaire, axis=1)
    tab2_somme = np.sum(v2_scalaire, axis=1)

    im1_ord[:,0] = tab1_somme
    im2_ord[:,0] = tab2_somme

    ################
    #Stock code RGB
    im1_ord[:,1:4] = im1_rgb
    im2_ord[:,1:4] = im2_rgb

    #################
    #Stock les indices
    #Je ne crée qu'un seul tableau d'indice car les images sont de même taille
    index = np.arange(len(im1_rgb))
    
    im1_ord[:,4] = index
    im2_ord[:,4] = index


    #Trier les 2 listes
    im1_ord = im1_ord[im1_ord[:,0].argsort()]
    im2_ord = im2_ord[im2_ord[:,0].argsort()]

    #Stocker les indices de l'image 1 dans l'image 2
    im2_ord[:,4:] = im1_ord[:,4:]

    

    return(im1_ord, im2_ord)


def cost_numpy(im1_rgb, im2_rgb, u):
    """ Entrée : deux listes de tuples et un tuple
        Sortie : un entier
        Calcul le coût entre deux listes rgb selon un vecteur u """

    im12_ord = to_ord_numpy(im1_rgb, im2_rgb, u)
    im1_ord = im12_ord[0]
    im2_ord = im12_ord[1]
    dist = 0.0

    calc = (im1_ord - im2_ord)**2
    dist = calc.sum()

    return dist


def best_cost_numpy(im1_rgb, im2_rgb):
    """ Entrée : deux listes de tuples
        Sortie : un entier
        Calcul le meilleur coût entre deux listes rgb """

    global best_u

    u = (randint(-500,500), randint(-500,500), randint(-500,500)) # rand en flottant !
    best_cost = cost_numpy(im1_rgb, im2_rgb, u)


    nb_u = 20 # nombre de tirages de u
    cost_list = [] # liste de tout les coûts que l'on va calculer
    for i in range(0, nb_u):
        u = (randint(-500,500), randint(-500,500), randint(-500,500))
        cost_cur = cost_numpy(im1_rgb, im2_rgb, u)
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

    im1_rgb = to_rgb_numpy(im1)
    im2_rgb = to_rgb_numpy(im2)

    best_cost_numpy(im1_rgb, im2_rgb)

    # on récuperre les deux listes ordonnées par le meilleur vercteur
    var = to_ord_numpy(im1_rgb, im2_rgb, best_u)
    im1_ord = var[0]
    im2_ord = var[1]
    im3_rgb = [0]*len(im1_ord)

    
    im2_ord = im2_ord[im2_ord[:,4].argsort()]

    im3_rgb = im2_ord[:,1:4]
    im3 = Image.new(im1.mode, im1.size)
    #On transforme le tableau en tuple afin de pouvoir l'enregistrer avec im.putdata
    im3_bis = tuple(map(tuple, im3_rgb))
    im3.putdata(im3_bis)
    
    print()
    print("sauvegarde du nouveau fichier")
    im3.save(fichier3, "BMP")
    print("Fin")

    im1.close()
    im2.close()

def main_numpy(fichier1, fichier2, fichier3):
    if intro(fichier1, fichier2):
        new_image_numpy(fichier1, fichier2, fichier3)
