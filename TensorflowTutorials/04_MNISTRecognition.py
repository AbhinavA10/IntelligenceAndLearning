#ABHINAV AGRAHARI
#This file: we make our own tensorflow model using tensors, and tensorgraphs


#https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow
import tensorflow as tf
#for image manipulation
import numpy as np
from PIL import Image

#importing the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded
#one-hot encoding means the binary vector of [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] represents the number 3
n_train = mnist.train.num_examples # 55,000
#n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000

#Neural Network now,
#number of neurons in each layer:
n_input = 784   # input layer (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 10   # output layer (0-9 digits)

#setting constants (hyperparams)
learning_rate = 1e-4 #how much paramters adjust at each step
n_iterations = 1000 #how many times we go through the training step,
batch_size = 128 #how many training examples we are using at each step
dropout = 0.5 # we use dropout in the final hidden layer to give each neuron in it a 50% chance of being eliminated every training step: helps prevent overfitting

#To build the network now, we setup as a tensorflow graph
X = tf.placeholder("float", [None, n_input]) #[undefined number of images, 784pixels]
Y = tf.placeholder("float", [None, n_output]) #[undefined number of outputs, 10 possible classes]
keep_prob = tf.placeholder(tf.float32) #3rd tensor, to control dropout rate: during training:0.5, during testing :1

#during training, the weight and bias values are updated
weights = { #set to rand near 0
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
biases = { #set to a single number initially
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}
#setting up layers of the network
#each hidden layer does matrix multiplication on previous layer's outputs and current layer's weights, and then addes the bias
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob) #last hidden layer applies the drouput rate of 0.5
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#we have to now define the loss function we want to optimize
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer)) #loss function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #type of gradient descent optimization

#Training the net
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1)) # we compare the output of our net, and the Y labels from mnist
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Propagate values forward through the network
# Compute the loss
# Propagate values backward through the network
# Update the parameters

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})
    # print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
print("\nAccuracy on test set:", test_accuracy)


imgName="all";
while imgName != "none":
    imgName = input("Enter name of file to test:  ");
    img2Fix = Image.open("customHandwrittenImages/"+imgName).convert('L');
    img2Fix = img2Fix.resize((28,28))
    img2Array = np.invert(img2Fix)
    img = img2Array.ravel()

    #img = np.invert(Image.open("test_img.png").convert('L')).ravel()
    #open function of Image lib loas image as 4D array: RGB, and alpha
    #we convert to grayscale with L
    #store this as a numpy array and invert bc .convert(L) makes black 0 and white 255. But mnist has it oppisite
    #ravel() flattens the array (unstacks)
    prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
    print ("Prediction for test image:", np.squeeze(prediction)) #np.squeeze converts [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] to the number 3
