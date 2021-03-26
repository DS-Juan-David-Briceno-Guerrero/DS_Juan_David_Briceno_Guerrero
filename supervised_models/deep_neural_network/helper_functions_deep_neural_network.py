#Implementation d'un reseau de neuron avec plusieurs des couches dans le hidden part. 
import inspect
import numpy as np
import matplotlib.pyplot as plt
#from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from help_shallow_neural_networks import plot_decision_boundary
#from help_shallow_neural_networks import sigmoid
#from help_shallow_neural_networks import initialize_with_zeros

#Fonctions qui aident a l'implementation du forwardpropagation.
#1.Initialization des paramettres W et b pour un reseaux de L couches.
def initialize_parameters(n_x, n_h, n_y):
  
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#layer dims cet un vecteur qui contient les nombre des neurons pour chacun couche du reseau de neurons.
def initialize_parameters_deep(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)          

    for l in range(1, L):
       
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
       
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def linear_forward(A, W, b):
    
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache


#2,Creation de la fonction d'activation qui calcule pour une couche l du reseu son activation et cache en correspondence avec la activation assigne..
# Le cache s'est les valeur de l'output Z.
def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
       
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#3.Implementation d'une fonction qui calcule tous les activations et les caches de chacun couche du reseau.
def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        
        A, cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

#4.Implementation de la fonction cost.

def compute_cost(AL, Y):
    
    m = Y.shape[1]
  
    cost = -(1/m)*(np.dot(Y,np.log(AL).T)+np.dot(1-Y,np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

#Fonctions qui aident a l'implementation du backwardpropagation.
#5.Calcule des valuers pour les gradients dW,db, et dA^(l-1) dans un couche l du reseaux,
def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis =1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

#Pour essayer l'implementation de la fonction est correcte.
dZ = np.array([[ 1.62434536 ,-0.61175641, -0.52817175 ,-1.07296862],
 [ 0.86540763 ,-2.3015387  , 1.74481176, -0.7612069 ],
 [ 0.3190391 , -0.24937038,  1.46210794 ,-2.06014071]]) 

linear_cache = (np.array([[-0.3224172 , -0.38405435,  1.13376944, -1.09989127],
       [-0.17242821, -0.87785842,  0.04221375,  0.58281521],
       [-1.10061918,  1.14472371,  0.90159072,  0.50249434],
       [ 0.90085595, -0.68372786, -0.12289023, -0.93576943],
       [-0.26788808,  0.53035547, -0.69166075, -0.39675353]]), np.array([[-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035],
       [ 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896],
       [-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548]]), np.array([[ 2.10025514],
       [ 0.12015895],
       [ 0.61720311]]))

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


#6.
def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


#7.Implementation d'un fonction qui fait le calcule des derivees dA^(l-1), dW, db en relation une fonction d'activation passee par parametre.
def linear_activation_backward(dA, cache, activation):
  
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
        
    elif activation == "sigmoid":
        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
    return dA_prev, dW, db



#8.Implementation d'un fonction qui fait le calcul des gradients dA, dW, db pour chacun couche du reseau de neurons.
def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+ str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads

#Pour essayer que l'implementation de la fonction L_model_backward c'est ok.
#caches contient les information de A_prev, W, B (linear_activation), et Z (activation_cache) pour chacon couche de reseau.
#La function qui calcule les grads A_L (couche finale de reseau) de la fonction de cost J est:
#dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)).
#Pour chacun couche du reseau il y a un calcule des grads dA_prev(output dans backpropagation), dW, db.
caches = (((np.array([[ 0.09649747, -1.8634927 ],
       [-0.2773882 , -0.35475898],
       [-0.08274148, -0.62700068],
       [-0.04381817, -0.47721803]]), np.array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
       [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
       [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]), np.array([[ 1.48614836],
       [ 0.23671627],
       [-1.02378514]])), np.array([[-0.7129932 ,  0.62524497],
       [-0.16051336, -0.76883635],
       [-0.23003072,  0.74505627]])), ((np.array([[ 1.97611078, -1.24412333],
       [-0.62641691, -0.80376609],
       [-2.41908317, -0.92379202]]), np.array([[-1.02387576,  1.12397796, -0.13191423]]), np.array([[-1.62328545]])), np.array([[ 0.64667545, -0.35627076]])))

Y_assess = np.array([[1, 0]])

AL = np.array([[ 1.78862847 , 0.43650985]])

grads = L_model_backward(AL, Y_assess, caches)

def print_grads(grads):
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dA1 = "+ str(grads["dA1"])) 
    
print_grads(grads)

#9.Implementation d'un fonction qui fait la mise a jour des parametres W, b pour chacun couche du reseau de neurons.
# W := W - alpha*dW
# b := b -alpha*db
def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+ str(l+1)]
    return parameters

