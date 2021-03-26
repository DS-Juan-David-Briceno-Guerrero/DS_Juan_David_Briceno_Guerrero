#TP d'apprentissage pour en savoir plus sur tensorflow, et son utilisation.
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import cv2

#%matplotlib inline
np.random.seed(1)
#Writing and running programs in TensorFlow has the following steps:
#1.Create Tensors (variables) that are not yet executed/evaluated.
#2.Write operations between those Tensors.
#3.Initialize your Tensors.
#4.Create a Session.

#Run the Session. This will run the operations you'd written above.
#Therefore, when we created a variable for the loss, we simply defined the loss as a function of other quantities, but did not evaluate its value. To evaluate it, we had to run init=tf.global_variables_initializer(). That initialized the loss variable, and in the last line we were finally able to evaluate the value of loss and print its value.

#Now let us look at an easy example. Run the cell below:


#1.Pour creer les tensor et ecrire un graphe computationel.

#Implementation d'une fonction.
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))  

#implementation d'un operation sur tensorflow.
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)

#Pour le lancer, il faut apres creer un session.
sess = tf.Session()
print(sess.run(c))

#Placeholders.
# Change the value of x in the feed_dict
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

#When you first defined x you did not have to specify a value for it. 
#A placeholder is simply a variable that you will assign data to only later, 
#when running the session. 
#We say that you feed data to these placeholders when running the session.

#Here's what's happening: When you specify the operations needed for a computation, 
#you are telling TensorFlow how to construct a computation graph. 
#The computation graph can have some placeholders whose values you will specify only later. 
#Finally, when you run the session, you are telling TensorFlow to execute the computation graph.

#2.Une fonction linaire.
def linear_function():
    
    np.random.seed(1)
    
    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X),b)
    ### END CODE HERE ### 
    
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    
    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ### 
    
    # close the session 
    sess.close()

    return result

print( "result = \n" + str(linear_function()))


#3.Implementation de la fonction de perte.
def cost(logits, labels):
    ### START CODE HERE ### 
    
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32, name = "z")
    a = tf.placeholder(tf.float32, name = "a")
    
    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,  labels = a)
    
    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()
    
    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict = {z:logits, a:labels})
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return cost

logits = np.array([0.2,0.4,0.7,0.9])

cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))

#4.Implementation de one-hot enconding avec tensorflow.
#Quand nous avons import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import cv2e besoin de convertir un vecteur de classification comme y=[1,2,3,0,2,1] a :
# y= [ 000100
#      100001
#      010010     
#      001000].

#Implementer une function pour creer des un's ou zeros quand nous utilison tensorflow.
def one_hot_matrix(labels, C):
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C,name ="C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels,C)
    
    # Create the session (approx. 1 line)
    sess = tf.Session() 
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 lilogits = np.array([0.2,0.4,0.7,0.9])
    sess.close()
    ### END CODE HERE ###
    
    return one_hots

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = \n" + str(one_hot))

def ones(shape):    
    ### START CODE HERE ###
    
    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
    ones = tf.ones(shape)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    return ones

print ("ones = " + str(ones([3])))


#5.Developpement d'un reseau de neuron en utilisant tensorflow.
#charger les donnees du echantillon SIGNS.
#Charger les donnes pour faire le train.
with h5py.File('/home/juan-david/Documents/data_science/travail_personnel/machine_learning/train_signs.h5') as hdf:
    ls = list(hdf.keys())
    print('List of databases in this file :\n', ls)
    data1 = hdf.get('train_set_x')
    data2 = hdf.get('train_set_y')
    dataset1 = np.array(data1)
    dataset2 = np.array(data2)
    dataset2 = dataset2.reshape(dataset2.shape[0],1)
    print('Shape of dataset1: \n', dataset1.shape)
    print('Shape of dataset2: \n', dataset2.shape)

print(dataset1)
print(dataset2)

#Charger les donnes pour faire le test.
with h5py.File('/home/juan-david/Documents/data_science/travail_personnel/machine_learning/test_signs.h5') as hdf:
    ls = list(hdf.keys())
    print('List of databases in this file :\n', ls)
    data3 = hdf.get('test_set_x')
    data4 = hdf.get('test_set_y')
    dataset3 = np.array(data3)
    dataset4 = np.array(data4)
    dataset4 = dataset4.reshape(dataset4.shape[0],1)
    print('Shape of dataset3: \n', dataset3.shape)
    print('Shape of dataset4: \n', dataset4.shape)

print(dataset3)
print(dataset4)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = dataset1, dataset2, dataset3, dataset4 
classes = np.array([0 ,1 ,2, 3 ,4 ,5])
#classes = np.array(['non-cat', 'cat'])
#classes = classes.reshape(1,classes.shape[0])

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


def create_placeholders(n_x, n_y):
    
    ### START CODE HERE ### (approx. 2 lines)
    X =  tf.placeholder(tf.float32, [n_x, None])
    Y =  tf.placeholder(tf.float32, [n_y, None])
    ### END CODE HERE ###
    
    return X, Y

X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))

#6.Initialization des parametrres. Il est fait avec hardcoding.
#L'initilization est choisi Xabier pour les poids du reseau.
#L'initilization est choisi Zero pour les biases.
def initialize_parameters():

    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


#7.Implementation de la forwardpropagation du reseau.
def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                      # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                      # Z3 = np.dot(W3, A2) + b3
    ### END CODE HERE ###
    
    return Z3

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))

#8.Implementation de la fonction de cost.
# GRADED FUNCTION: compute_cost 
def compute_cost(Z3, Y):
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    ### END CODE HERE ###
    
    return cost


#9.Implementation du backwardpropagation dans un model complet qui reunis tous les fonctions implementés.
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters




#Afin de tester le performance du modele, nous pouvons ajouter une nouvelle photo et faire la prediction.
#import scipy
#from PIL import Image
#from scipy import ndimage

## START CODE HERE ## (PUT YOUR IMAGE NAME) 
#my_image = "3-avec-la-main.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
#fname = "images/" + my_image
#image = np.array(ndimage.imread(fname, flatten=False))
#image = image/255.
#my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
#my_image_prediction = predict(my_image, parameters)

#plt.imshow(image)
#print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))






