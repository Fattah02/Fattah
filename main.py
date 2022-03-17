import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from math import e
from tqdm import tqdm
#from utilities import *


#on genere un dataset X y, 100 lignes et 2 var
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0]), 1)


print('dimensions de X = ',X.shape)
print('dimensions de y = ', y.shape)
#X.shape donne la dimension de X
#X.shape[0] = ligne


#on s'aide de la matrice X pour donner au vecteur une dimension => autant de parametres w que de variables x
def initialisation(X):
    W = np.random.randn(X.shape[1],1) #2 parametres car 2 var, 2L 1C
    b = np.random.randn(1) #reel
    return (W,b) #tuple

# on a les dimensions de w et b (les parametres)


def model(X,W,b):
    #produit matriciel entre X et W, d'après les calculs sur feuille
    Z = X.dot(W) + b #fonction linéaire avec parametres et variables
    A = 1 / (1 + np.exp(-Z)) #fonction d'activation (de réponse) qui retourne vecteur 100L 1C
    return A

# on a 100 valeurs dans notre échantillon donc 100 activations

#m = nombre d'échantillons, de points dans y
# on cherche les distances entre les resultats et les données qu'on avait
# quantifie les erreurs du modèle, c'est un réel
def log_loss(A,y):
    m = len(y) #nombre d'echnatillons, de points dans m
    epsilon = 1e-15 #eviter d'avoir log(0)
    return 1/m * np.sum(-y * np.log(A+epsilon) - (1-y) * np.log(1-A+epsilon))

def gradients(A,X,y):
    m = len(y)
    dW = 1/m * np.dot(X.T, A-y) #dl/dw = jacobien
    # dw = X.t = transposee, produit avec A-y
    db = 1/m * np.sum(A-y)
    return (dW,db)

def update(dW,db, W, b, learning_step):
    #implementation formules gradient
    W -= learning_step * dW
    b -= learning_step * db
    return (W,b)



def predict(X, W, b):
    A = model(X,W,b)
    #print(A)
    return A >= 0.5


# on a le modèle, on peut maintenant faire des prédictions
# en gros maintenant si on a une plante avec la largeur et la longueur et qu'on la rentre dans le modèle
# on aura une probabilité de la toxicité ou non
# sortie du modèle = probabilité > 0,5 = toxique = y1

# Implementons cela

# on rassemble maintenant tout cela
# pour créer la boucle d'apprentissage

def artificial_neuron(X,y, learning_step= 0.1, n_cycle=100):
    # initialiser W et b
    W, b = initialisation(X)
    #on voit l'évolution du cout :
    Loss = []
    acc = [] # performance du modèle

    for i in tqdm(range(n_cycle)):

        #ACtivation
        A = model(X,W,b) #vecteur A

        if i % 10 == 0:

            #calcul du cout
            Loss.append(log_loss(A,y)) #evolution de l'erreur,cout

            #accuracy
            y_pred = predict(X,W,b)
            acc.append(accuracy_score(y, y_pred))

        #MAJ
        dW, db = gradients(A,X,y)
        W, b = update(dW,db,W,b,learning_step)
        #W et b sont mis a jour et réutilisés pour refaire des prédictions
        # qui seront recomparés et ainsi recalculer les gradients et remettre a jour...

    # comparaison des données de ref y et nos prédictions
    #print(accuracy_score(y, y_pred))
    #pourcentage de bonne réponse

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(Loss) #graphiquement voir l'évo
    plt.subplot(1,2,2)
    plt.plot(acc)
    plt.show()

    return (W,b)
# Usine à modeles







#overflow = valeur trop grande de l'exponentielle
#normaliser = mettre sur une meme echelle les variables du dataset













#EN 3D
fig = go.Figure(data=[go.Scatter3d(
    x=X[:,0].flatten(),
    y=X[:,1].flatten(),
    z=y.flatten(),
    mode = 'markers',
    marker = dict(
        size = 5,
        color=y.flatten(),
        colorscale='YlGn',
        opacity=0.8,
        reversescale=True
    )

)])


fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()