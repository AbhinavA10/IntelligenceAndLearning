#ABHINAV AGRAHARI
#Tensorflow's basic classification tutorial, using the fashion_mnist dataset

#https://www.tensorflow.org/tutorials/keras/basic_classification
#https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
#This guide trains a neural network model to classify images of clothing, like sneakers and shirts.

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

#fashion.mnist is the hello world of ML
fashion_mnist = keras.datasets.fashion_mnist

#train_images and train_labels arrays are the training set—the data the model uses to learn.
#The model is tested against the test set, the test_images, and test_labels arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#images are actually 28x28 NumPy arrays

#labels are integer numbers corresponding to the following classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape) # prints (60000, 28, 28) meaning 60000 images, and each one is 28x28 pixels
print(len(train_labels)) # there are 60000 labels
print(train_labels) #shows an array of ints
print(test_images.shape)
print(len(test_labels))

#From the below, we see that the pixel values in this image range from 0 to 255
##plt.figure()
##plt.imshow(train_images[0])
##plt.colorbar()
##plt.grid(False)
##plt.show() #imshow doesnt actually show it on the screen
#to scale the pixels to 0 or 1, we divide by 255. AKA normalizing the data

#also, we need to pre-process the training set and the testing set in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

#Now we test if our labeling is right so far, by displaying the first 25 images
#in the training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()  
#Now, we build the model. think of a neural layer as the vertical thing in 3blue1brown's video
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Dense(128, activation=tf.nn.relu), #sequence of fully-connected neural layer, this one has 128 nodes
    keras.layers.Dense(10, activation=tf.nn.softmax) # a softmax layer returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
])
model.compile(optimizer=tf.train.AdamOptimizer(), ##  how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy', #This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction
              metrics=['accuracy']) #Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified

#Training the model
# To train a model, we need to feed the model data, let it learn, then compare its predictions about a test set to  the expected result
model.fit(train_images, train_labels, epochs=5)
#seeing how the model performs on the test data set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# since the accuracy on the test data is less than the training data --> overfitting
# Overfitting: is when a machine learning model performs worse on new data than on their training data

#Since model is trained, we make some predictions about images
predictions = model.predict(test_images) #here, prediction is an array of 10 numbers (for the 10 labels) aka the model's confidence that the image is of that type
print(predictions[0]) #print the prediction for the first one
#using the numpy library, we can find the max confidence
print(np.argmax(predictions[0]))
#and then compare this to the expected output
print(test_labels[0])

#graphing this to look at the full set of the 10 channels
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
#plotting the first 15 images now
i=15
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

#to make a predicition about a single image, instead of all of them as before
  # Grab an image from the test dataset
img = test_images[0]

print(img.shape)
# Add the image to a batch where it's the only member, bc tf.keras is optimzied to make predictions on a batch
img = (np.expand_dims(img,0))

print(img.shape)
predictions_single = model.predict(img)

print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
