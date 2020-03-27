import tensorflow as tf
# print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt

# plt.imshow(training_images[0]) #
# print(training_labels[0])
# print(training_images[0])

training_images = training_images / 255.0 #one set for training our model
test_images = test_images / 255.0 #one set for testing whether our model works or not

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation = tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation = tf.nn.softmax)])
# Sequential: defines a sequence of layers in the neural network
#Flatten: takes a "square(umm...)" and turns it into a 1 dimensional set.
#Dense: Adds a layer of neurons
#Relu: Only passes values 0 or greater to the next layer in the network
#Softmax: takes a set of values and picks the biggest one
         #input:[0.1, 0,1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05]
         #output: [0,0,0,0,1,0,0,0,0]

model.compile(optimizer = tf.optimizers.Adam(),
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])
model.fit(training_images, training_labels, epochs = 5)

model.evaluate(test_images, test_labels)
