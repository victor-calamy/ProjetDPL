from platform import java_ver
import pandas as pd
import numpy as np
import scipy
import librosa 
import random
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Flatten,Dropout 
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
    
#Hyperparamètres
class hyperparams : 
    couches =[] #liste pour le nombre de neurones par couches denses
    dropout = [] #valeur du dropout pour chaque couches
    fonctionAct=[] #fonction activation pour chaque couches
    epoques = 0
    batchsize = 5

    def toString(self):
        return("couches : " + str(self.couches) + " dropout : " + str(self.dropout) + " fonctionAct : " + str(self.fonctionAct) + " epoques : " + str(self.epoques))


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
    for k in range(len(Y)):
        num = LISTE_LABELS.index(Y[k][0])
        Y[k][0]=num
    Y=np.array(Y)

    return( train_test_split(X, Y, test_size=0.1))


"""----------------------------------------------------------------------------------------------------------------------

Création du modèle 

----------------------------------------------------------------------------------------------------------------------"""
def createModelDense(hyperparams):
    model = Sequential()
    model.add(Flatten())# on applatit les données

    for k in range (len(hyperparams.couches)):
            model.add(Dense(hyperparams.couches[k],activation = hyperparams.fonctionAct[k]))
            if(hyperparams.dropout[k] !=0):
                model.add(Dropout(hyperparams.dropout[k]))

    model.add(Dense(NB_CLASSES, activation='softmax'))#couche de sortie

    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['acc'])

    return(model)

def entrainement(hyperparams):
    x_train, x_test, y_train, y_test = createDataset()
    y_train = to_categorical(y_train, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    model = createModelDense(hyperparams)
    res = model.fit(x_train,
          y_train,
          batch_size=5,
          epochs=hyperparams.epoques)

    #validation :
    return(model.evaluate(x_test,y_test),res)
hyperparam1 = hyperparams()
hyperparam1.couches = [256,128,64,32,16]
hyperparam1.dropout = [0.2,0.1,0,0,0]
hyperparam1.fonctionAct = ["relu","tanh","sigmoid","tanh","sigmoid"]
hyperparam1.epoques = 50
entrainement(hyperparam1)

def testPlusieursHyperParam():
    """
    Cette fonction nous permet de tester une palette d'hyperparametres afin de déterminer les meilleurs. On va prendre 3 configurations par hyperparam.

    """
    ListeCouches = [[32],[64],[128]]
    listeDropout = [[0],[0.1],[0.2]]
    listeFonAct= [["relu"],["tanh"],["sigmoid"]]
    liste_hyperparams = []
    for k in range (len(ListeCouches)):
        for i in range (len(listeDropout)):
            for j in range (len(listeFonAct)):
                hyperparam1 = hyperparams()
                hyperparam1.couches = ListeCouches[k]
                hyperparam1.dropout = listeDropout[i]
                hyperparam1.fonctionAct = listeFonAct[j]
                hyperparam1.epoques = 20
                liste_hyperparams.append(hyperparam1)

    liste_resultats=[]
    for k in range  (len(liste_hyperparams)):
        eval, acc = entrainement(liste_hyperparams[k])
        liste_resultats.append([liste_hyperparams[k].toString(),[eval[1], acc.history['acc'][-1]]])
    return(liste_resultats)

