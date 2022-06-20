import pandas as pd
import numpy as np
import scipy
import librosa 
import random
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Flatten 
import matplotlib.pyplot as plt
from os import listdir
from keras.utils import to_categorical
from playsound import playsound
import multiprocessing
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""----------------------------------------------------------------------------------------------------------------------

Liste de fonctions et de variables utiles

----------------------------------------------------------------------------------------------------------------------"""
DOSSIER_AUDIOS = "C:/Users/Victor Calamy/Desktop/DPL/projetDPL/audios/"
LISTE_AUDIOS = [f for f in listdir(DOSSIER_AUDIOS)]
LISTE_LABELS = ["Meow","Motorcycle","Traffic_noise_and_roadway_noise","Child_speech_and_kid_speaking","Church_bell","Bicycle_bell","Walk_and_footsteps"]
NB_CLASSES = len(LISTE_LABELS)

#_______________________________________________ Affichage _______________________________________
def affichage(img) : 
    plt.figure(figsize=(15,10))
    plt.title("Spectrogramme de l'audio", weight='bold')
    plt.imshow(img)
    plt.show()
    



"""----------------------------------------------------------------------------------------------------------------------

Avant de commencer à créer notre modèle, il faut réfléchir à la représentation des données. Nous avons plusieurs possibilités
        - étudier le spectrogramme

----------------------------------------------------------------------------------------------------------------------"""


def getSpectrogramme (file_name):
    """
    cette fonction retourne le spectrogramme de l'audio "path"
    """
    path = DOSSIER_AUDIOS + file_name
    duree = 5
    taux_echantillonage = 44100
    nb_data = duree * taux_echantillonage
 
    y, _ = librosa.core.load(path, sr=taux_echantillonage) # Charge l'audio avec un taux d'échantillonnage de 44 100 Hz
    y,_ = librosa.effects.trim(y) #On enlève les silences de l'audio
    if (len(y)>nb_data):
        y   = y[0:nb_data] # on met nos données à la bonne taille 
    else:
        decalage = (nb_data - len(y)) // 2
        y = np.pad(y, (decalage, nb_data - len(y) - decalage), 'constant') # on rajoute des 0 au début et à la fin si notre audio n'est pas assez long

    #On va maintenant transformer cet echantillonage en un spectrogramme 

    spectrogram = librosa.feature.melspectrogram(y, sr=taux_echantillonage) 
    spectrogram = librosa.power_to_db(spectrogram).astype(np.float32) # On passe en dB et en float pour la suite

    #Enfin, on normalise les donées 
    img = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

    return(img)

def createDataset():
    """
    Création du dataset de spectrogrammes avec leur label. On split aussi les données en un jeu de test et un jeu d'entrainement 
    """
    loadY = pd.read_csv("C:/Users/Victor Calamy/Desktop/DPL/projetDPL/train_curated.csv",sep=',')
    X = []
    Y=[]
    for audio in LISTE_AUDIOS:
        X.append(getSpectrogramme(audio).transpose())
        index=loadY[loadY["fname"] == audio].index
        Y.append(loadY["labels"][index].values)
    X=np.array(X)

    # maintenant, on va remplacer les label string en entier.
    print(np.unique(np.array(Y)))
    for k in range(len(Y)):
        num = LISTE_LABELS.index(Y[k][0])
        Y[k][0]=num
    Y=np.array(Y)

    return( train_test_split(X, Y, test_size=0.1))


"""----------------------------------------------------------------------------------------------------------------------

Création du modèle :

----------------------------------------------------------------------------------------------------------------------"""
    

def createModelDense():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=['acc'])

    return(model)

def entrainement():
    x_train, x_test, y_train, y_test = createDataset()
    y_train = to_categorical(y_train, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    model = createModelDense()
    model.fit(x_train,
          y_train,
          batch_size=5,
          epochs=100)

    #validation :
    model.evaluate(x_test,y_test)
entrainement()
